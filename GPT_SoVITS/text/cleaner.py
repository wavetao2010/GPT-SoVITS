from text import chinese, cleaned_text_to_sequence, symbols, english
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
from phonemizer.punctuation import Punctuation
language_module_map = {"zh": chinese, "en": english}
special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]


def clean_text(text, language):
    if(language not in language_module_map):
        language="en"
        text=" "
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol)
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    # 初始化分隔符
    if language == "zh":
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
        ipas = []
    else:
        separator = Separator(word="_", syllable="-", phone="|")

        # 创建espeak后端实例
        phonemizer = EspeakBackend(
            language='en-us' if language == "en" else language,
            punctuation_marks=Punctuation.default_marks(),
            preserve_punctuation=True,
            with_stress=False,
            tie=False,
            language_switch="keep-flags",
            words_mismatch="ignore",
        )
        phones = language_module.g2p(norm_text)
        norm_text2ipa = norm_text.split()
        ipas = phonemizer.phonemize(
            norm_text2ipa,
            separator=separator,
            strip=True,
            njobs=1
        )
        ipas = [ipa.split('|') for ipa in ipas]
        word2ph = [len(word) for word in ipas]
        ipas = ['1' + ipa for sublist in ipas for ipa in sublist]
    for ph in phones:
        assert ph in symbols
    return phones, word2ph, norm_text,ipas


def clean_special(text, language, special_s, target_symbol):
    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


def text_to_sequence(text, language):
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones)


if __name__ == "__main__":
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))
