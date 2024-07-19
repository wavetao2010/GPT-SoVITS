import json
from pathlib import Path
import re
import os
import regex
import torch
import torch.nn as nn
import torchaudio as ta
from dataclasses import dataclass
import numpy as np
import librosa
import string
@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

@dataclass
class Segment:
    label: str
    start: float
    end: float

    def __repr__(self):
        return f"{self.label}: [{self.start:.2f}, {self.end:.2f})"

    @property
    def length(self):
        return self.end - self.start

class PhoneAligner():
    def __init__(self, pt_file_path, sample_rate, transcript):
        super(PhoneAligner, self).__init__()

        self.frame_shit = 240
        self.sample_rate = float(sample_rate)
        self.model = torch.jit.load(pt_file_path).eval().cuda()
        self.mfcc = ta.transforms.MFCC(sample_rate=sample_rate, n_mfcc=13, melkwargs={
                               'win_length': 600, 'hop_length': self.frame_shit, 'n_fft': 1024, 'n_mels': 40, 'f_min': 0.0, 'f_max': 8000.0}).cuda()
        self.transcript = {idx + 1: token for idx, token in transcript.items()}
        self.transcript[0] = '<blank>'

    @torch.inference_mode()
    def __call__(
            self, 
            x, # (T)
            y # 
            ) -> list:
        with torch.no_grad():
            x = x + 1
            transcript = {idx: self.transcript[token] for idx, token in enumerate(x.tolist())}
            y = self.model(self.mfcc(y)).squeeze(0).cpu()
            trellis = self.get_trellis(y, x)
            path = self.backtrack(trellis, y, x)
            segments = self.merge_repeats(path, transcript)

            return segments


    def get_trellis(self, emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        trellis = torch.zeros((num_frame, num_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[-num_tokens + 1 :, 0] = float("inf")

        for t in range(num_frame - 1):
            trellis[t + 1, 1:] = torch.maximum(
                trellis[t, 1:] + emission[t, blank_id],
                trellis[t, :-1] + emission[t, tokens[1:]],
            )
        return trellis
    
    def backtrack(self, trellis, emission, tokens, blank_id=0):
        t, j = trellis.size(0) - 1, trellis.size(1) - 1

        path = [Point(j, t, emission[t, blank_id].exp().item())]
        while j > 0:
            assert t > 0

            p_stay = emission[t - 1, blank_id]
            p_change = emission[t - 1, tokens[j]]
            stayed = trellis[t - 1, j] + p_stay
            changed = trellis[t - 1, j - 1] + p_change
            t -= 1
            if changed > stayed:
                j -= 1
            prob = (p_change if changed > stayed else p_stay).exp().item()
            path.append(Point(j, t, prob))
        while t > 0:
            prob = emission[t - 1, blank_id].exp().item()
            path.append(Point(j, t - 1, prob))
            t -= 1

        return path[::-1]
    
    def merge_repeats(self, path, transcript):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index * self.frame_shit / self.sample_rate,
                    (path[i2 - 1].time_index + 1) * self.frame_shit / self.sample_rate,
                )
            )
            i1 = i2
        return segments

class Aligner():
    def __init__(self) -> None:
        
        self.gsv2token = json.load(open(f'{str(Path.cwd())}/GPT_SoVITS/phone.json', 'r'))
        self.idx2token = {idx: self.gsv2token[token] for idx, token in enumerate(list(self.gsv2token.keys()))}
        self.gsv_phones = list(self.gsv2token.keys())
        self.model = PhoneAligner(f'{str(os.getcwd())}/GPT_SoVITS/pretrained_models/gsv_phone.pt', 24000, self.idx2token)

    def __call__(self, norm_text, word2phone, phones, audio, sr, align_offset_time):
        audio = audio.astype(np.float32)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        audio = librosa.util.normalize(audio)
        audio, _= librosa.effects.trim(audio, top_db=40)
        audio = torch.from_numpy(audio).unsqueeze(0).cuda()
        phones = phones.copy()
        norm_text = regex.findall(r'\p{Han}|[a-zA-Z]+|\p{P}', norm_text)
        print('对齐前的文本:',norm_text)
        assert len(word2phone) == len(norm_text)
        assert sum(word2phone) == len(phones)

        start_pos = 0
        text2index = []
        for i in range(len(word2phone)):
            text2index.append([start_pos, start_pos + (word2phone[i] - 1)])
            start_pos += word2phone[i]

        # for i, index in enumerate(text2index):
        #     print(norm_text[i], phones[index[0]:index[1] + 1])

        for i in range(len(phones)):
            phones[i] = phones[i].lower()
            if 'v' in phones[i]:
                phones[i] = phones[i].replace('v', 'u')
            if phones[i] not in self.gsv_phones:
                if 'ir' in phones[i]:
                    phones[i] = phones[i].replace('ir', 'i')
                elif 'i0' in phones[i]:
                    phones[i] = phones[i].replace('i0', 'i')
                elif '5' in phones[i]:
                    phones[i] = phones[i].replace('5', '1')
                elif not re.search(r'[a-z]', phones[i]):
                    phones[i] = '_'
                elif re.match(r'[a-z]+', phones[i]):
                    phones[i] = phones[i + 1]
                if phones[i] not in self.gsv_phones:
                    print('phone not in gsv2token: ', phones[i], )
                    phones[i] = '_'

        phones_align = []
        phones2phones_align_index = []
        last_phone = None
        for i in range(len(phones)):
            if phones[i] != last_phone:
                if phones[i] != '_' or i != len(phones) - 1:
                    phones_align.append(self.gsv_phones.index(phones[i]))
                    last_phone = phones[i]

            phones2phones_align_index.append(len(phones_align) - 1)
        segments = self.model(torch.tensor(phones_align), audio)

        def is_punctuation(s):
            return all(c in string.punctuation for c in s)

        text_timestamps = []
        for i, text in enumerate(norm_text):
            text_type = 'punctuation' if is_punctuation(text) else 'word'
            text_timestamps.append({
                'seg': text,
                'start': segments[phones2phones_align_index[text2index[i][0]]].start + align_offset_time,
                'end': segments[phones2phones_align_index[text2index[i][1]]].end + align_offset_time,
                'boundary_type': text_type
            })
        return text_timestamps



