# -*- coding: utf-8 -*-
import json
import torch
import allspark
import os
import sys
import re
from pathlib import Path
import regex
import torch.nn as nn
import torchaudio as ta
from dataclasses import dataclass
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import signal
from time import time as ttime
import librosa
import soundfile as sf
import uvicorn
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from GPT_SoVITS.feature_extractor import cnhubert
from io import BytesIO
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence,sequence_to_cleaned_text, symbols
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
import config as global_config
import base64
import traceback
import LangSegment
#导入初始化参数
g_config = global_config.Config()
# model_name = "TVB"
# sovits_path = g_config.sovits_path[model_name]
# gpt_path = g_config.gpt_path[model_name]
# refer_wav_path = g_config.refer_path[model_name]
# prompt_text = g_config.refer_text[model_name]
# prompt_language = g_config.refer_language[model_name]
cnhubert_base_path = g_config.cnhubert_path
cnhubert.cnhubert_base_path = cnhubert_base_path
bert_path = g_config.bert_path
dtype = torch.float32
dict_language = {
    "中文": "zh",
    "英文": "en",
    "日文": "ja",
    "ZH": "zh",
    "EN": "en",
    "JA": "ja",
    "zh": "zh",
    "en": "en",
    "ja": "ja"
}
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)


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
            'win_length': 600, 'hop_length': self.frame_shit, 'n_fft': 1024, 'n_mels': 40, 'f_min': 0.0,
            'f_max': 8000.0}).cuda()
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
        trellis[-num_tokens + 1:, 0] = float("inf")

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



class MyProcessor(allspark.BaseProcessor):
    """ MyProcessor is a example
        you can send mesage like this to predict
        curl -v http://127.0.0.1:8080/api/predict/service_name -d '2 105'
    """

    def initialize(self):
        """ load module, executed once at the start of the service
             do service intialization and load models in this function.
        """
        allspark.default_properties().put('rpc.keepalive', '10000000')
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
        bert_model = bert_model.to(device)
        ssl_model = cnhubert.get_model()
        ssl_model = ssl_model.to(device)
        self.gsv2token = json.load(open(f'{str(Path.cwd())}/GPT_SoVITS/phone.json', 'r'))
        self.idx2token = {idx: self.gsv2token[token] for idx, token in enumerate(list(self.gsv2token.keys()))}
        self.gsv_phones = list(self.gsv2token.keys())
        aligner_model = PhoneAligner(f'{str(os.getcwd())}/GPT_SoVITS/pretrained_models/gsv_phone.pt', 24000,
                                  self.idx2token)

        model_names = g_config.model_names_list
        self.model_dicts = {}
        for model_name in model_names:
            models = []
            gpt_path = g_config.gpt_path[model_name]
            sovits_path = g_config.sovits_path[model_name]
            t2s_model = self.change_gpt_weights(gpt_path)
            vq_model, hps = self.change_sovits_weights(sovits_path)
            models.append(vq_model)
            models.append(t2s_model)
            models.append(hps)
            self.model_dicts[model_name] = models

        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.ssl_model = ssl_model
        self.aligner_model = aligner_model

    def pre_process(self, data):
        json_data = json.loads(data)
        model_name = json_data.get("model_name")
        text = json_data.get("text")
        text_language = json_data.get("text_language")
        return model_name,text, text_language

    def post_process(self, data):
        """ process after process
        """
        audio_base64 = data.get("audio")
        align_info = data.get("align_info")
        response_data = {
            "audio": audio_base64,
            "align_info": align_info,
        }
        json_response = json.dumps(response_data, ensure_ascii=False)
        return str(json_response).encode('utf-8')

    def process(self, data):
        try:
            model_name,text, text_language = self.pre_process(data)
            refer_wav_path = g_config.refer_path[model_name]
            prompt_text = g_config.refer_text[model_name]
            prompt_language = g_config.refer_language[model_name]
            with torch.no_grad():
                gen = self.get_tts_wav(
                    model_name,refer_wav_path, prompt_text, prompt_language, text, text_language
                )
                sampling_rate, audio_data, align_info = next(gen)

            # # save test align info
            # json.dump(align_info, open("./tmp/align_info.json", "w", encoding="utf-8"), ensure_ascii=False)

            wav = BytesIO()
            sf.write(wav, audio_data, sampling_rate, format="wav")
            wav.seek(0)

            # Convert audio stream to base64
            audio_base64 = base64.b64encode(wav.read()).decode('utf-8')
            data = {
                "audio": audio_base64,
                "align_info": align_info
            }
            torch.cuda.empty_cache()
            if self.device == "mps":
                print('executed torch.mps.empty_cache()')
                torch.mps.empty_cache()
            # Return audio stream and timestamp info as JSON
            return self.post_process(data), 200
        except:
            print(traceback.format_exc())
            error_result = {
                "status_text": 'error',
                "status_code": '400',
                "message": traceback.format_exc()}
            return str(error_result).encode('utf-8'), 400
    def change_gpt_weights(self,gpt_path):
        global hz, max_sec, t2s_model, config
        hz = 50
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        config = dict_s1["config"]
        max_sec = config["data"]["max_sec"]
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        t2s_model = t2s_model.to(self.device)
        t2s_model.eval()
        return t2s_model
    def change_sovits_weights(self,sovits_path):

        dict_s2 = torch.load(sovits_path, map_location="cpu")
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        vq_model = vq_model.to(self.device)
        vq_model.eval()
        return vq_model, hps

    def get_first(self, text):
        pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
        text = re.split(pattern, text)[0].strip()
        return text

    def aligner(self, norm_text, word2phone, phones, audio, sr, align_offset_time):
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
        segments = self.aligner_model(torch.tensor(phones_align), audio)

        text_timestamps = []
        for i, text in enumerate(norm_text):
            text_timestamps.append({
                'seg': text,
                'start': segments[phones2phones_align_index[text2index[i][0]]].start + align_offset_time,
                'end': segments[phones2phones_align_index[text2index[i][1]]].end + align_offset_time
            })
        return text_timestamps


    def clean_text_inf(self,text, language):
        phones, word2ph, norm_text,ipa = clean_text(text, language)
        phones = cleaned_text_to_sequence(phones)
        return phones, word2ph, norm_text,ipa

    def get_bert_inf(self,phones, word2ph, norm_text, language):
        language = language.replace("all_", "")
        if language == "zh":
            bert = self.get_bert_feature(norm_text, word2ph).to(self.device)  # .to(dtype)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32,
            ).to(self.device)

        return bert

    def get_bert_feature(self,text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)  #####输入是long不用管精度问题，精度随bert_model
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        # if(is_half==True):phone_level_feature=phone_level_feature.half()
        return phone_level_feature.T

    def get_phones_and_bert(self,text, language):
        if language in {"en", "all_zh", "all_ja"}:
            language = language.replace("all_", "")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 因无法区别中日文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            phones_sequence, word2ph, norm_text, ipa = self.clean_text_inf(formattext, language)
            if language == "zh":
                bert = get_bert_feature(norm_text, word2ph).to(self.device)
            else:
                bert = torch.zeros(
                    (1024, len(phones_sequence)),
                    dtype=torch.float32,
                ).to(self.device)
        elif language in {"zh", "ja", "auto"}:
            textlist = []
            langlist = []
            LangSegment.setfilters(["zh", "ja", "en", "ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "ko":
                        langlist.append("zh")
                        textlist.append(tmp["text"])
                    else:
                        langlist.append(tmp["lang"])
                        textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            print(textlist)
            print(langlist)
            phones_sequence_list = []
            bert_list = []
            norm_text_list = []
            phones_list = []
            word2ph_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                # 获取音素ID序列，word2ph，规范化文本，音标列表
                phones_sequence, word2ph, norm_text, ipa = self.clean_text_inf(textlist[i], lang)
                # 获取bert特征
                bert = self.get_bert_inf(phones_sequence, word2ph, norm_text, lang)
                if lang == "zh":
                    phone = sequence_to_cleaned_text(phones_sequence)
                    phones_list.append(phone)
                else:
                    phones_list.append(ipa)
                phones_sequence_list.append(phones_sequence)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
                word2ph_list.append(word2ph)
            bert = torch.cat(bert_list, dim=1)
            phones_sequence = sum(phones_sequence_list, [])
            phones = sum(phones_list, [])
            word2ph = sum(word2ph_list, [])
            norm_text = ''.join(norm_text_list)
        return phones_sequence, bert.to(dtype), norm_text, phones, word2ph

    def get_spepc(self,hps, filename):
        audio = load_audio(filename, int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                                 hps.data.win_length, center=False)
        return spec

    def merge_short_text_in_array(self,texts, threshold):
        if (len(texts)) < 2:
            return texts
        result = []
        text = ""
        for ele in texts:
            text += ele
            if len(text) >= threshold:
                result.append(text)
                text = ""
        if (len(text) > 0):
            if len(result) == 0:
                result.append(text)
            else:
                result[len(result) - 1] += text
        return result

    def get_tts_wav(self,model_name,ref_wav_path, prompt_text, prompt_language, text, text_language, top_k=5, top_p=1, temperature=1):
        prompt_language = dict_language[prompt_language]
        text_language = dict_language[text_language]
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        prompt_language, text = prompt_language, text.strip("\n")
        text = text.strip("\n")
        if (text[0] not in splits and len(
            self.get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text
        model_array = self.model_dicts[model_name]
        vq_model = model_array[0]
        t2s_model = model_array[1]
        hps = model_array[2]
        zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float32)
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            wav16k = wav16k.to(self.device)
            zero_wav_torch = zero_wav_torch.to(self.device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = self.ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = vq_model.extract_latent(ssl_content)

            prompt_semantic = codes[0, 0]
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        texts = text.split("\n")
        texts = self.merge_short_text_in_array(texts, 5)
        audio_opt = []
        align_offset_time = 0
        align_infos = []
        phones1, bert1, norm_text1, _, _ = self.get_phones_and_bert(prompt_text, prompt_language)
        for text in texts:
            phones2, bert2, norm_text2_align, phones2_align, word2ph2_align = self.get_phones_and_bert(text, text_language)
            print(f"参考文本维度：{bert1.shape}, TTS文本维度：{bert2.shape}")
            bert = torch.cat([bert1, bert2], 1)
            print(f"合并后维度：{bert.shape}")
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
            prompt = prompt_semantic.unsqueeze(0).to(self.device)
            with torch.no_grad():
                # pred_semantic = t2s_model.model.infer(
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec,
                )
            # print(pred_semantic.shape,idx)
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)  # .unsqueeze(0)#mq要多unsqueeze一次
            refer = self.get_spepc(hps, ref_wav_path)  # .to(device)
            refer = refer.to(self.device)
            audio = (
                    vq_model.decode(
                    pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0), refer
                )
                .detach()
                .cpu()
                .numpy()[0, 0]
            )
            max_audio = np.abs(audio).max()  # 简单防止16bit爆音
            if max_audio > 1: audio /= max_audio
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            align_infos += self.aligner(norm_text2_align, word2ph2_align, phones2_align,
                                   np.concatenate([audio, zero_wav], 0), hps.data.sampling_rate, align_offset_time)
            align_offset_time += np.concatenate([audio, zero_wav], 0).size / hps.data.sampling_rate
        yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16), align_infos




if __name__ == '__main__':
    # parameter worker_threads indicates concurrency of processing
    runner = MyProcessor(worker_threads=20, endpoint="0.0.0.0:8001")
    runner.run()