"""Build the offline conversation pronunciation lexicon using misaki==0.9.4.

Run in the dictionary TTS preparation venv, after prepare_dictionary_tts.py.
Derived from Misaki's Apache-2.0 US lexicons; defaults select the common reading.
Unknown words are spelled at runtime. No Python or eSpeak runtime is shipped.
"""
import gzip
import argparse
import importlib.metadata
import json
from pathlib import Path
from misaki.en import Lexicon

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'app/src/main/assets/tts'
parser = argparse.ArgumentParser()
parser.add_argument('--work-dir', type=Path, default=Path('/tmp/tts-models'))
args = parser.parse_args()
lexicon = Lexicon(False)
entries = {}
for source in (lexicon.silvers, lexicon.golds):
    for word, phones in source.items():
        if not all(c.isascii() and (c.isalpha() or c == "'") for c in word):
            continue
        if isinstance(phones, dict):
            phones = phones.get('DEFAULT')
        if phones and (word.lower() not in entries or word.islower()):
            entries[word.lower()] = phones.replace('ɾ', 'T').replace('ʔ', 't')
# AAPT strips .gz extensions and decompresses those assets; .bin preserves the gzip container.
with gzip.GzipFile(ASSETS / 'english-lexicon.bin', 'wb', mtime=0) as output:
    output.write(''.join(f'{word}\t{phones}\n' for word, phones in sorted(entries.items())).encode())
config = json.loads((args.work_dir / 'kokoro-config.json').read_text())
(ASSETS / 'kokoro-vocab.json').write_text(json.dumps(config['vocab'], ensure_ascii=False))
dist = importlib.metadata.distribution('misaki')
license_file = next(p for p in dist.files if str(p).endswith('licenses/LICENSE'))
(ASSETS / 'licenses/Misaki-LICENSE.txt').write_text(dist.locate_file(license_file).read_text())
(ASSETS / 'licenses/Misaki-NOTICE.txt').write_text(
    f"Pronunciation lexicon derived from Misaki {dist.version} (https://github.com/hexgrad/misaki).\n"
    "Modified: merged US gold/silver lexicons, selected default readings, lowercased keys, "
    "normalized flap/glottal symbols, and compressed as TSV.\n")
print(f'Prepared {len(entries)} offline pronunciations.')
