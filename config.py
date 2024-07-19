import sys,os

import torch

# 推理用的指定模型
sovits_path = {
    "xiong2": "SoVITS_weights/xiong2_e24_s792.pth",
    "wukong": "SoVITS_weights/wukong_e24_s840.pth",
    "silang": "SoVITS_weights/silang_e20_s460.pth",
    "TVB": "SoVITS_weights/TVB_e24_s504.pth",
    "maikease":"SoVITS_weights/maikease_e24_s1152.pth",
    "paimeng":"SoVITS_weights/paimeng2_e110_s159940.pth"
}
gpt_path = {
    "xiong2": "GPT_weights/xiong2-e15.ckpt",
    "wukong": "GPT_weights/wukong-e15.ckpt",
    "silang": "GPT_weights/silang-e15.ckpt",
    "TVB": "GPT_weights/TVB-e15.ckpt",
    "maikease":"GPT_weights/maikease-e15.ckpt",
    "paimeng":"GPT_weights/paimeng2-e100.ckpt"
}

refer_path = {
    "xiong2": "refer/xiong2.wav",
    "wukong": "refer/wukong.wav",
    "silang": "refer/silang.wav",
    "TVB": "refer/TVB.wav",
    "maikease":"refer/maikease.wav"
}

prompt_text = {
    "xiong2": "支持多种语言，包括中文、英语、日语、韩语、法语、德语、西班牙语、阿拉伯语等50多种语言。",
    "wukong": "当贝播放器是一款专注大屏端的本地播放软件，专为智能电视盒子投影打造。",
    "silang": "最近我们店里在疯狂地做着一款生日蛋糕，安妮贝壳蛋糕店，环境超级干净。",
    "TVB": "可以 FB 先开,平滑过去,不过 FB 没谷歌稳定,所以可以把预算分下,谷歌占小部分",
    "maikease":"支持多种语言，包括中文、英语、日语、韩语、法语、德语、西班牙语、阿拉伯语等50多种语言。"
}

prompt_language = {
    "xiong2": "zh",
    "wukong": "zh",
    "silang": "zh",
    "TVB": "zh",
    "maikease":"zh"
}

emotion_list = {
    "xiong2": {
        "default": {
            "refer_path": "refer/xiong2.wav",
            "prompt_text": "支持多种语言，包括中文、英语、日语、韩语、法语、德语、西班牙语、阿拉伯语等50多种语言。",
            "prompt_language": "zh"
        }},
    "wukong": {
        "default": {
            "refer_path": "refer/wukong.wav",
            "prompt_text": "当贝播放器是一款专注大屏端的本地播放软件，专为智能电视盒子投影打造。",
            "prompt_language": "zh"
        }
    },
    "silang": {
        "default": {
            "refer_path": "refer/silang.wav",
            "prompt_text": "最近我们店里在疯狂地做着一款生日蛋糕，安妮贝壳蛋糕店，环境超级干净。",
            "prompt_language": "zh"
        }
    },
    "TVB": {
        "default": {
            "refer_path": "refer/TVB.wav",
            "prompt_text": "可以 FB 先开,平滑过去,不过 FB 没谷歌稳定,所以可以把预算分下,谷歌占小部分",
            "prompt_language": "zh"
        }
    },
    "maikease":{
        "default": {
            "refer_path": "refer/maikease.wav",
            "prompt_text": "支持多种语言，包括中文、英语、日语、韩语、法语、德语、西班牙语、阿拉伯语等50多种语言。",
            "prompt_language": "zh"
        }
    },
    "paimeng": {
        "default": {
            "refer_path": "refer/说话—既然罗莎莉亚说足迹上有元素力，用元素视野应该能很清楚地看到吧。.wav",
            "prompt_text": "既然罗莎莉亚说足迹上有元素力，用元素视野应该能很清楚地看到吧。",
            "prompt_language": "zh"
        },
        "angry": {
            "refer_path": "refer/生气—呜哇好生气啊！不要把我跟一斗相提并论！.wav",
            "prompt_text": "呜哇好生气啊！不要把我跟一斗相提并论！",
            "prompt_language": "中文"
        },
        "excited": {
            "refer_path": "refer/激动—好耶！《特尔克西的奇幻历险》出发咯！.wav",
            "prompt_text": "好耶！《特尔克西的奇幻历险》出发咯！",
            "prompt_language": "中文"
        },
        "empathetic": {
            "refer_path": "refer/疑问—哇，这个，还有这个…只是和史莱姆打了一场，就有这么多结论吗？.wav",
            "prompt_text": "哇，这个，还有这个…只是和史莱姆打了一场，就有这么多结论吗？",
            "prompt_language": "中文"
        }
    }
}

model_names_list = ['xiong2', 'wukong', 'silang', 'TVB', 'maikease']




is_half_str = os.environ.get("is_half", "True")
is_half = True if is_half_str.lower() == 'true' else False
is_share_str = os.environ.get("is_share","False")
is_share= True if is_share_str.lower() == 'true' else False

cnhubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
pretrained_sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"

exp_root = "logs"
python_exec = sys.executable or "python"
if torch.cuda.is_available():
    infer_device = "cuda"
else:
    infer_device = "cpu"

webui_port_main = 9874
webui_port_uvr5 = 9873
webui_port_infer_tts = 9872
webui_port_subfix = 9871

api_port = 9880

if infer_device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    if (
            ("16" in gpu_name and "V100" not in gpu_name.upper())
            or "P40" in gpu_name.upper()
            or "P10" in gpu_name.upper()
            or "1060" in gpu_name
            or "1070" in gpu_name
            or "1080" in gpu_name
            or "A100" in gpu_name.upper()
            or "A10" in gpu_name.upper()
    ):
        is_half=False

if(infer_device=="cpu"):is_half=False

class Config:
    def __init__(self):
        self.sovits_path = sovits_path
        self.gpt_path = gpt_path
        self.is_half = is_half

        self.cnhubert_path = cnhubert_path
        self.bert_path = bert_path
        self.pretrained_sovits_path = pretrained_sovits_path
        self.pretrained_gpt_path = pretrained_gpt_path

        self.exp_root = exp_root
        self.python_exec = python_exec
        self.infer_device = infer_device

        self.refer_path = refer_path
        self.refer_text = prompt_text
        self.refer_language = prompt_language
        self.emotion_list = emotion_list
        self.model_names_list = model_names_list

        self.webui_port_main = webui_port_main
        self.webui_port_uvr5 = webui_port_uvr5
        self.webui_port_infer_tts = webui_port_infer_tts
        self.webui_port_subfix = webui_port_subfix

        self.api_port = api_port
