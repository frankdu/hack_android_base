#!/usr/bin/env python3
"""Make compact test-only BPE fixtures from exported, full tokenizer specs.

The reduced vocab/merges support only the included prompts, never app generation.
"""
import argparse
import html
import json
from pathlib import Path
import unicodedata
import regex


def compact(spec):
    vocab, merges = spec['vocab'], spec['merges']
    ranks = {pair: i for i, pair in enumerate(merges)}
    selected, words = set(), set()
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = {n: chr(n) for n in byte_values}
    extra = 0
    for n in range(256):
        if n not in chars:
            chars[n] = chr(256 + extra); extra += 1
    specials = {s['text'] for s in spec['specialTokens']}
    special = regex.compile('|'.join(regex.escape(s) for s in sorted(specials, key=len, reverse=True)))
    pattern = regex.compile(spec.get('pattern') or r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+")
    for f in spec['fixtures']:
        text = unicodedata.normalize('NFC', spec['prefix'] + f['prompt'] + spec['suffix'])
        if spec['kind'] == 'clip':
            text = regex.sub(r'\s+', ' ', html.unescape(html.unescape(text))).strip().lower()
        for segment in special.split(text):
            for word in pattern.findall(segment):
                parts = [chars[b] for b in word.encode()]
                if spec['kind'] == 'clip': parts[-1] += '</w>'
                while len(parts) > 1:
                    candidates = [(ranks.get(a + ' ' + b, float('inf')), a, b) for a, b in zip(parts, parts[1:])]
                    rank, first, second = min(candidates)
                    if rank == float('inf'): break
                    selected.add(first + ' ' + second)
                    merged, i = [], 0
                    while i < len(parts):
                        if i + 1 < len(parts) and parts[i] == first and parts[i + 1] == second:
                            merged.append(first + second); i += 2
                        else:
                            merged.append(parts[i]); i += 1
                    parts = merged
                words.update(parts)
    words.update(specials)
    return dict(spec, vocab={word: vocab[word] for word in sorted(words)},
                merges=[pair for pair in merges if pair in selected], testOnly=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(compact(json.loads(a.spec.read_text())), ensure_ascii=False, indent=2) + '\n')
