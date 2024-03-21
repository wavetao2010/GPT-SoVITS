import argparse
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import traceback
import requests
from glob import glob
import soundfile as sf
import librosa
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tools.asr.config import check_w_local_models
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

language_code_list = [
    "af", "am", "ar", "as", "az", 
    "ba", "be", "bg", "bn", "bo", 
    "br", "bs", "ca", "cs", "cy", 
    "da", "de", "el", "en", "es", 
    "et", "eu", "fa", "fi", "fo", 
    "fr", "gl", "gu", "ha", "haw", 
    "he", "hi", "hr", "ht", "hu", 
    "hy", "id", "is", "it", "ja", 
    "jw", "ka", "kk", "km", "kn", 
    "ko", "la", "lb", "ln", "lo", 
    "lt", "lv", "mg", "mi", "mk", 
    "ml", "mn", "mr", "ms", "mt", 
    "my", "ne", "nl", "nn", "no", 
    "oc", "pa", "pl", "ps", "pt", 
    "ro", "ru", "sa", "sd", "si", 
    "sk", "sl", "sn", "so", "sq", 
    "sr", "su", "sv", "sw", "ta", 
    "te", "tg", "th", "tk", "tl", 
    "tr", "tt", "uk", "ur", "uz", 
    "vi", "yi", "yo", "zh", "yue",
    "auto"]

def execute_asr(input_folder, output_folder, model_size, language,precision):
    if '-local' in model_size:
        model_size = model_size[:-6]
        model_path = f'tools/asr/models/whisper-{model_size}'
    else:
        model_path = model_size
    if language == 'auto':
        language = 'zh' #不设置语种由模型自动输出概率最高的语种
    print("loading whisper model:",model_size,model_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_path)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=False,
            torch_dtype=torch_dtype,
            device=device,
        )
    except:
        return print(traceback.format_exc())
    output = []
    output_file_name = os.path.basename(input_folder)
    output_file_path = os.path.abspath(f'{output_folder}/{output_file_name}.list')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in tqdm(glob(os.path.join(input_folder, '**/*.wav'), recursive=True)):
        try:
            data, sampling_rate = sf.read(file)
            if sampling_rate != 16000:
                data = librosa.resample(y=data, orig_sr=sampling_rate, target_sr=16000)
                sampling_rate = 16000
            # input_features = processor(data, sampling_rate=sampling_rate,
            #                            return_tensors="pt").input_features
            # predicted_ids = model.generate(input_features)
            # text = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            result = pipe(data)
            text = result['text']
            output.append(f"{file}|{output_file_name}|{language.upper()}|{text}")
        except:
            return print(traceback.format_exc())
        
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
        print(f"ASR 任务完成->标注文件路径: {output_file_path}\n")
    return output_file_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", type=str, required=True,
                        help="Path to the folder containing WAV files.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, 
                        help="Output folder to store transcriptions.")
    parser.add_argument("-s", "--model_size", type=str, default='large-v3', 
                        choices=check_w_local_models(),
                        help="Model Size of Faster Whisper")
    parser.add_argument("-l", "--language", type=str, default='ja',
                        choices=language_code_list,
                        help="Language of the audio files.")
    parser.add_argument("-p", "--precision", type=str, default='float16', choices=['float16','float32'],
                        help="fp16 or fp32")

    cmd = parser.parse_args()
    output_file_path = execute_asr(
        input_folder  = cmd.input_folder,
        output_folder = cmd.output_folder,
        model_size    = cmd.model_size,
        language      = cmd.language,
        precision     = cmd.precision,
    )
