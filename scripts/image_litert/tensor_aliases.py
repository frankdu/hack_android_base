"""Read exact signature tensor identities; shape equality is not proof of aliasing."""
from pathlib import Path
from ai_edge_litert import schema_py_generated as schema


def signature_aliases(path, key):
    data=Path(path).read_bytes(); model=schema.Model.GetRootAsModel(data,0)
    for i in range(model.SignatureDefsLength()):
        signature=model.SignatureDefs(i)
        if signature.SignatureKey().decode()!=key: continue
        inputs={signature.Inputs(j).TensorIndex():signature.Inputs(j).Name().decode()
            for j in range(signature.InputsLength())}
        return {signature.Outputs(j).Name().decode():inputs[signature.Outputs(j).TensorIndex()]
            for j in range(signature.OutputsLength()) if signature.Outputs(j).TensorIndex() in inputs}
    raise ValueError(f'Missing signature {key}: {path}')


def annotate_manifest(root, manifest):
    root=Path(root); aliases={}
    for action in manifest['actions']:
        key=action['model'],action['signature']
        if key not in aliases: aliases[key]=signature_aliases(root/action['model'],action['signature'])
        for output in action['outputs']:
            output.pop('aliasOfInput',None)
            if output['name'] in aliases[key]: output['aliasOfInput']=aliases[key][output['name']]
    return sum(len(v) for v in aliases.values())
