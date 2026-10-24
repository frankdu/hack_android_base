#!/usr/bin/env python3
"""Refresh checksums and validate the exact staged public file sets before upload."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import ModelCard


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def finalize(root):
    root=Path(root); plans=[]
    for stage in sorted((root/'publications').iterdir()):
        if not stage.is_dir(): continue
        previous=json.loads((stage/'artifact_manifest.json').read_text())
        card=ModelCard((stage/'README.md').read_text())
        assert card.data.pipeline_tag=='text-to-image'
        assert card.data.license in ('other','apache-2.0')
        files={}
        for p in sorted(stage.rglob('*')):
            if not p.is_file(): continue
            name=str(p.relative_to(stage))
            if name in ('artifact_manifest.json','SHA256SUMS'): continue
            assert not any(part in ('.cache','.git','compiler-work') for part in p.relative_to(stage).parts)
            assert p.resolve().is_relative_to(stage.resolve())
            files[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
        for name,spec in previous['files'].items():
            # Only our fixed standalone SD runner may differ from the initial packaging pass.
            if name!='run_portable.py': assert files[name]==spec,(stage.name,name,'changed')
        previous['files']=files
        (stage/'artifact_manifest.json').write_text(json.dumps(previous,indent=2)+'\n')
        files['artifact_manifest.json']=dict(bytes=(stage/'artifact_manifest.json').stat().st_size,sha256=digest(stage/'artifact_manifest.json'))
        (stage/'SHA256SUMS').write_text(''.join(f'{s["sha256"]}  {name}\n' for name,s in sorted(files.items())))
        files['SHA256SUMS']=dict(bytes=(stage/'SHA256SUMS').stat().st_size,sha256=digest(stage/'SHA256SUMS'))
        plans.append(dict(repository=previous['repository'],directory=str(stage),files=files,bytes=sum(s['bytes'] for s in files.values())))
        print('VERIFIED',previous['repository'],len(files),'files',sum(s['bytes'] for s in files.values()),flush=True)
    assert len(plans)==5
    (root/'publication-plan.json').write_text(json.dumps(plans,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args();finalize(a.root)
