#!/usr/bin/env python3
"""Audit pinned Android file URLs and mirror equivalence against live Hub metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from huggingface_hub import HfApi, hf_hub_download

PATTERN = re.compile(r'ImageFile\("([^"]+)", "([^"]+)", "([^"]+)", "([^"]+)", (\d+)L, "([^"]+)"(?:, listOf\(ImageLocation\("([^"]+)", "([^"]+)", "([^"]+)"\)\))?\)')


def audit(catalog, cache):
    text=Path(catalog).read_text(); api=HfApi(); metadata={}; checked=[]
    for match in PATTERN.finditer(text):
        filename,repo,rev,path,size,sha,mirror,mrev,mpath=match.groups()
        if filename.startswith('IRIS_LITERT/'): continue
        for source,revision,remote in [(repo,rev,path)]+([(mirror,mrev,mpath)] if mirror else []):
            key=source,revision
            if key not in metadata:
                metadata[key]={f.rfilename:f for f in api.model_info(source,revision=revision,files_metadata=True).siblings}
            spec=metadata[key][remote]
            assert spec.size==int(size),(source,remote,'size')
            if spec.lfs: actual=spec.lfs.sha256
            else:
                local=hf_hub_download(source,remote,revision=revision,cache_dir=cache)
                actual=hashlib.sha256(Path(local).read_bytes()).hexdigest()
            assert actual==sha,(source,remote,'sha256')
            checked.append(dict(filename=filename,repository=source,revision=revision,path=remote,bytes=int(size),sha256=sha))
    assert len({x['filename'].split('/')[0] for x in checked})==4
    return dict(modelCount=4,fileCount=len({x['filename'] for x in checked}),sourceFileChecks=len(checked),
                sourceSnapshots=len(metadata),status='PASS',files=checked)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',default='app/src/main/java/dev/gemmabench/imagegen/CommunityImageModels.kt')
    p.add_argument('--cache',default='artifacts/image-litert/catalog-hf-cache');p.add_argument('--out',default='artifacts/image-litert/android-catalog-audit.json')
    a=p.parse_args();result=audit(a.catalog,a.cache);Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))
