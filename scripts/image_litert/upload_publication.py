#!/usr/bin/env python3
"""Upload explicitly prepared LiteRT public packages and verify the Hub artifacts.

Requires --upload to publish. Uses the existing Hugging Face login; never reads or
prints token contents. Only names listed in publication-plan.json are uploaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError


def publish(root, namespace, upload):
    root=Path(root); plans=json.loads((root/'publication-plan.json').read_text())
    assert len(plans)==5
    api=HfApi(); assert api.whoami()['name']==namespace
    results=[]
    for plan in plans:
        repo=plan['repository']; directory=Path(plan['directory'])
        assert repo.startswith(namespace+'/') and directory.resolve().is_relative_to((root/'publications').resolve())
        assert set(plan['files'])=={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and '.cache' not in p.relative_to(directory).parts}
        print('PUBLICATION',repo,len(plan['files']),'files',plan['bytes'],'bytes',flush=True)
        if not upload: continue
        # Existing repositories may only be empty/new or one of our own resumed publications.
        try:
            existing=api.model_info(repo,files_metadata=True)
            names={p.rfilename for p in existing.siblings}
            if names- {'.gitattributes'}:
                previous=hf_hub_download(repo,filename='artifact_manifest.json')
                assert json.loads(Path(previous).read_text())['repository']==repo
            assert not existing.private, 'Refusing to change visibility of an existing private repository'
        except RepositoryNotFoundError:
            subprocess.run(['hf','repos','create',repo,'--public','--format','json'],check=True)
        # Exact whitelist avoids uploading caches or unrelated workstation content.
        commit=api.upload_folder(repo_id=repo,folder_path=str(directory),
            allow_patterns=list(plan['files']),
            commit_message='Publish verified LiteRT text-to-image pipeline and CPU examples')
        info=api.model_info(repo,revision=commit.oid,files_metadata=True,token=False)
        assert not info.private and info.gated is False
        remote={f.rfilename:f for f in info.siblings}
        for name,spec in plan['files'].items():
            f=remote[name]
            assert f.size==spec['bytes'],(repo,name,'size')
            if f.lfs:
                assert f.lfs.sha256==spec['sha256'],(repo,name,'sha256')
            else:
                data=(directory/name).read_bytes()
                git_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                assert git_sha==f.blob_id,(repo,name,'git blob')
        graph=next(name for name in plan['files'] if name.endswith('.tflite'))
        public=get_hf_file_metadata(hf_hub_url(repo,graph,revision=info.sha),token=False)
        assert public.size==plan['files'][graph]['bytes']
        result=dict(repository=repo,url='https://huggingface.co/'+repo,revision=info.sha,
                    public=True,gated=False,verifiedFiles=len(plan['files']),bytes=plan['bytes'],
                    allFileSizesAndHashesMatch=True,anonymousGraphDownloadVerified=True)
        results.append(result)
        (root/'publication-results.json').write_text(json.dumps(results,indent=2)+'\n')
        print('PUBLISHED_AND_VERIFIED',json.dumps(result),flush=True)
    if not upload: print('Dry run only; use --upload to publish.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--namespace',required=True)
    p.add_argument('--upload',action='store_true');a=p.parse_args();publish(a.root,a.namespace,a.upload)
