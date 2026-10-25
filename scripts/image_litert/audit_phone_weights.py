#!/usr/bin/env python3
"""Inspect stored constant dtypes in every neural graph of a compact SD bundle."""
import argparse
import json
from pathlib import Path
from collections import Counter

from ai_edge_litert import schema_py_generated as schema


def audit(root):
    root=Path(root); manifest=json.loads((root/'manifest.json').read_text())
    names={v:k for k,v in vars(schema.TensorType).items() if isinstance(v,int)}
    graphs=[]
    for name in manifest['files']:
        if not name.startswith(('graphs/clip','graphs/unet','graphs/vae')): continue
        data=(root/name).read_bytes(); model=schema.Model.GetRootAsModel(data,0)
        counts=Counter(); seen=set()
        for index in range(model.SubgraphsLength()):
            graph=model.Subgraphs(index)
            for i in range(graph.TensorsLength()):
                tensor=graph.Tensors(i); buffer=tensor.Buffer()
                stored=model.Buffers(buffer)
                length=stored.DataLength() or stored.Size()  # LiteRT external-buffer layout
                if buffer not in seen and length>1024:
                    counts[names[tensor.Type()]]+=length; seen.add(buffer)
        required='FLOAT16' if name.startswith('graphs/vae') else 'INT8'
        assert counts[required]>0,(name,dict(counts))
        graphs.append(dict(path=name,storedConstantBytesByDtype=dict(counts)))
    result=dict(status='PASS',modelId=manifest['modelId'],graphs=graphs,
        note='Stored constant dtypes verify compression; they do not measure runtime memory or accelerator coverage.')
    (root/'weight-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(manifest['modelId'],len(graphs),'neural graphs:',result['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle');a=p.parse_args();audit(a.bundle)
