#!/usr/bin/env python3
"""Prepare pronunciation tokens and selected voice styles; models stay downloadable.

Python 3.12: pip install 'misaki[en]==0.9.4' phonemizer-fork espeakng-loader numpy
Usage: python scripts/prepare_dictionary_tts.py --work-dir /tmp/tts-models
The phonemizer is a build-time tool, never linked into the Android application.
"""
import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import espeakng_loader
import numpy as np
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.backend import EspeakBackend

EspeakWrapper.set_library(espeakng_loader.get_library_path())
import os
os.environ['ESPEAK_DATA_PATH'] = espeakng_loader.get_data_path()
from misaki import en, espeak

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'app/src/main/assets/tts'
KOKORO_REV = '1939ad2a8e416c0acfeecc08a694d14ef25f2231'
KITTEN_REV = '84781d74e29ee25217551556398b42f80593a813'
KOKORO_VOICE = 'am_adam'
KOKORO_VOICE_SHA256 = '162b035ed91cfc48b6046982184c645f72edcdd1b82843347f605d7bf7b15716'
KITTEN_VOICE_ID = 'expr-voice-5-f'  # Kiki; reference speed prior: 0.8.


def fetch(work, name, url):
    path = work / name
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    config = fetch(args.work_dir, 'kokoro-config.json',
                   'https://huggingface.co/hexgrad/Kokoro-82M/resolve/785407d1adfa7ae8fbef8ffd85f34ca127da3039/config.json')
    vocab = json.loads(config.read_text())['vocab']
    voice = fetch(args.work_dir, f'kokoro-{KOKORO_VOICE}.bin',
                  f'https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/{KOKORO_REV}/voices/{KOKORO_VOICE}.bin')98
    assert hashlib.sha256(voice.read_bytes()).hexdigest() == KOKORO_VOICE_SHA256
    (ASSETS / 'kokoro-voice.bin').write_bytes(voice.read_bytes())
    voices_path = fetch(args.work_dir, 'kitten-voices.npz',
                        f'https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8/resolve/{KITTEN_REV}/voices.npz')
    assert hashlib.sha256(voices_path.read_bytes()).hexdigest() == '8aa7cee235abb0739cb51e6559685f65a4dacd95568833d05699b1633f519b3f'
    voices = np.load(voices_path)[KITTEN_VOICE_ID]
    voices.astype('<f4').tofile(ASSETS / 'kitten-voice.bin')
    # Same symbol order and punctuation handling as KittenTTS 0.8 TextCleaner.
    symbols = '$' + ';:,.!?¡¿—…"«»"" ' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' + "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    kitten_vocab = {symbol: index for index, symbol in enumerate(symbols)}
    g2p = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False))
    kitten_g2p = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
    dictionary = (ROOT / 'seed/kids_english_dictionary.txt').read_bytes()
    words = list(dict.fromkeys(line.strip() for line in dictionary.decode().splitlines()
                              if line.strip() and not line.startswith('Category:')))
    output = {}
    for word in words:
        phones, _ = g2p(word)
        missing = set(phones) - vocab.keys()
        if missing:
            raise ValueError(f'Unsupported Kokoro phonemes for {word}: {missing}')
        text = word + ','  # Kitten adds a trailing comma for isolated words.
        kitten_phones = ' '.join(re.findall(r'\w+|[^\w\s]', kitten_g2p.phonemize([text])[0]))
        output[word] = {'kokoro': [vocab[c] for c in phones],
                        'kitten': [kitten_vocab[c] for c in kitten_phones if c in kitten_vocab],
                        'kittenStyle': min(len(text), voices.shape[0] - 1)}
    (ASSETS / 'dictionary-tokens.json').write_text(json.dumps({
        'dictionarySha256': hashlib.sha256(dictionary).hexdigest(), 'words': output}, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f'Prepared {len(words)} unique words. Kitten voice shape: {voices.shape}')


if __name__ == '__main__':
    main()
