#!/usr/bin/env python3
"""Prepare an identity-disjoint LFW calibration/holdout fixture; never ship photos.

Download https://ndownloader.figshare.com/files/5976015 to artifacts/lfw-funneled.tgz,
then run this script. Photos remain under ignored artifacts/people-validation/.
"""
import argparse
import hashlib
import json
import random
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, default=Path("artifacts/lfw-funneled.tgz"))
parser.add_argument("--output", type=Path, default=Path("artifacts/people-validation"))
args = parser.parse_args()
rng = random.Random(42)
args.output.mkdir(parents=True, exist_ok=True)
with tarfile.open(args.archive) as archive:
    identities = {}
    for member in archive.getmembers():
        if member.isfile() and member.name.endswith(".jpg"):
            identities.setdefault(Path(member.name).parent.name, []).append(member)
    identities = [(label, sorted(images, key=lambda m: m.name)) for label, images in sorted(identities.items()) if len(images) >= 8]
    rng.shuffle(identities)
    selected = identities[:40]
    assert len(selected) == 40
    records = []
    for i, (label, images) in enumerate(selected):
        rng.shuffle(images)
        for j, member in enumerate(images[:8]):
            name = f"person-{i:02d}-{j:02d}.jpg"
            data = archive.extractfile(member).read()
            (args.output / name).write_bytes(data)
            records.append(dict(file=name, identity=f"person-{i:02d}", split="calibration" if i < 20 else "holdout", sha256=hashlib.sha256(data).hexdigest(), source=member.name))
manifest = dict(dataset="LFW funneled; figshare file 5976015", seed=42, identitiesDisjoint=True, records=records)
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Prepared {len(records)} photos, 20 identities per split, 8 photos per identity.")
