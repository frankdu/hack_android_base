"""Static, weight-bounded FX shards and a tensor-only execution plan.

A shard keeps its weights as constants (never enormous runtime input tensors).
Only one CompiledModel is resident on Android at a time. No custom operators.
"""
import operator
from pathlib import Path
import re

import numpy as np
import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import tree_flatten, tree_unflatten


def attribute(module, path):
    value = module
    for part in path.split("."):
        value = getattr(value, part)
    return value


def static_shards(module, inputs, weight_limit):
    graph = torch.export.export(module.eval(), tuple(inputs), strict=False).module()
    # Export guards check the host's fixed shapes, not neural computations.
    for node in list(graph.graph.nodes):
        if node.op == "call_module" and str(node.target) == "_guards_fn" and not node.users:
            graph.graph.erase_node(node)
    graph.recompile()
    current, used, size = 0, set(), 0

    def partition(node):
        nonlocal current, used, size
        weights = [n for n in node.all_input_nodes if n.op == "get_attr"]
        incoming = {str(n.target): attribute(graph, str(n.target)) for n in weights if str(n.target) not in used}
        added = sum(t.numel() * t.element_size() for t in incoming.values() if isinstance(t, torch.Tensor))
        if size and size + added > weight_limit:
            current += 1
            used, size = set(), 0
            incoming = {str(n.target): attribute(graph, str(n.target)) for n in weights}
            added = sum(t.numel() * t.element_size() for t in incoming.values() if isinstance(t, torch.Tensor))
        used.update(incoming)
        size += added
        return current

    return split_module(graph, graph, partition, keep_original_order=True)


def bind_weights(child, arguments, parent_nodes):
    """Move split_module's lifted parameter arguments back into the shard."""
    root, graph, env, dynamic = torch.nn.Module(), Graph(), {}, []
    placeholders = [n for n in child.graph.nodes if n.op == "placeholder"]
    for index, (node, value, source) in enumerate(zip(placeholders, arguments, parent_nodes)):
        if isinstance(source, Node) and source.op == "get_attr":
            name = f"constant_{index}"
            if isinstance(value, torch.Tensor):
                root.register_buffer(name, value.detach())
            else:
                setattr(root, name, value)
            env[node] = graph.get_attr(name)
        else:
            env[node] = graph.placeholder(f"x{len(dynamic)}")
            dynamic.append(value)
    for node in child.graph.nodes:
        if node.op != "placeholder":
            env[node] = graph.node_copy(node, lambda n: env[n])
    return GraphModule(root, graph), dynamic


def natural_names(names):
    # LiteRT Torch's tuple input/output ABI. Fail rather than silently reorder tensors.
    for name in names:
        if not re.search(r"\d+$", name):
            raise ValueError(f"Unexpected converter tensor name {name}; inspect this LiteRT Torch version's ABI")
    return sorted(names, key=lambda name: int(re.search(r"\d+$", name)[0]))


class ProgramBuilder:
    def __init__(self, out, shard_mib=256, converter=None):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.limit = shard_mib * 1024 * 1024
        self.actions, self.externals, self.files = [], [], set()
        self.values, self.counter, self.recipes = {}, 0, {}
        self.converter = converter or self.convert

    def ref(self, value):
        name = f"t{self.counter}"
        self.counter += 1
        self.values[name] = value.detach().cpu().contiguous()
        return name

    def external(self, kind, value, **extra):
        ref = self.ref(value)
        self.externals.append(dict(tensor=ref, kind=kind, shape=list(value.shape), dtype=str(value.numpy().dtype), **extra))
        return ref

    def constant(self, value):
        ref = self.ref(value)
        relative = f"constants/{ref}.bin"
        target = self.out / relative
        target.parent.mkdir(exist_ok=True)
        self.values[ref].numpy().tofile(target)
        self.externals.append(dict(tensor=ref, kind="constant", path=relative, shape=list(value.shape), dtype=str(value.numpy().dtype)))
        self.files.add(relative)
        return ref

    def convert(self, module, arguments, path, expected):
        import litert_torch
        from ai_edge_litert.interpreter import Interpreter
        litert_torch.convert(module.eval(), tuple(arguments)).export(str(path))
        if path.stat().st_size >= 2_000_000_000:
            raise RuntimeError(f"{path}: graph exceeds 2GB; reduce --shard-mib or split a single large matrix")
        model = Interpreter(model_path=str(path), num_threads=4)
        model.allocate_tensors()
        signatures = model.get_signature_list()
        if len(signatures) != 1:
            raise RuntimeError(f"{path}: expected one signature, got {signatures}")
        signature = next(iter(signatures))
        names_in = natural_names(signatures[signature]["inputs"])
        names_out = natural_names(signatures[signature]["outputs"])
        if len(names_in) != len(arguments) or len(names_out) != len(expected):
            raise RuntimeError(f"{path}: converter ABI changed")
        actual = model.get_signature_runner(signature)(**dict(zip(names_in, [v.numpy() for v in arguments])))
        errors = []
        for name, want in zip(names_out, expected):
            got, want = actual[name], want.numpy()
            if got.shape != want.shape or not np.isfinite(got).all() or not np.allclose(got, want, rtol=0.01, atol=0.001):
                raise RuntimeError(f"{path}/{name}: CPU conversion parity failed; do not compile or publish")
            errors.append(float(np.max(np.abs(got - want))) if got.size else 0)
        return dict(signature=signature, inputNames=names_in, outputNames=names_out, maxAbsErrors=errors)

    def call(self, key, module, inputs, phase, step=0):
        args = [self.values[ref] for ref in inputs]
        if key in self.recipes:
            recipe = self.recipes[key]
            # Rebind a component plan to a new timestep without copying its weights.
            refs = dict(zip(recipe["inputs"], inputs))
            for action in recipe["actions"]:
                output_refs = [self.ref(value) for value in recipe["outputExamples"][action["model"]]]
                new = dict(action, inputs=[dict(i, tensor=refs[i["tensor"]]) for i in action["inputs"]],
                           outputs=[dict(o, tensor=r) for o, r in zip(action["outputs"], output_refs)], phase=phase, step=step)
                refs.update({o["tensor"]: r for o, r in zip(action["outputs"], output_refs)})
                self.actions.append(new)
            # Export-time numerical evaluation is still fresh, so scheduler updates get current values.
            with torch.no_grad():
                outputs, _ = tree_flatten(module(*args))
            result = [refs[r] for r in recipe["outputs"]]
            for ref, value in zip(result, outputs):
                self.values[ref] = value.detach().cpu().contiguous()
            return result
        parent = static_shards(module, args, self.limit)
        placeholders = iter(inputs)
        env, references, examples = {}, {}, {}
        start = len(self.actions)
        for node in parent.graph.nodes:
            if node.op == "placeholder":
                ref = next(placeholders)
                references[node], env[node] = ref, self.values[ref]
            elif node.op == "get_attr":
                env[node] = attribute(parent, str(node.target))
            elif node.op == "call_function" and node.target == operator.getitem:
                env[node] = env[node.args[0]][node.args[1]]
                references[node] = references[node.args[0]][node.args[1]]
            elif node.op == "call_module":
                arguments = [env[n] for n in node.args]
                child = parent.get_submodule(str(node.target))
                bound, dynamic = bind_weights(child, arguments, node.args)
                refs = [references[n] for n in node.args if n.op != "get_attr"]
                with torch.no_grad():
                    value = bound(*dynamic)
                flat, structure = tree_flatten(value)
                if not flat or any(not isinstance(v, torch.Tensor) for v in flat):
                    raise RuntimeError("Shard ABI must contain only tensors")
                filename = f"graphs/{key}-{node.target}.tflite"
                path = self.out / filename
                path.parent.mkdir(exist_ok=True)
                abi = self.converter(bound, dynamic, path, flat)
                outputs = [self.ref(v) for v in flat]
                examples[filename] = [v.detach().cpu().contiguous() for v in flat]
                self.actions.append(dict(model=filename, signature=abi["signature"],
                    inputs=[dict(name=n, tensor=r) for n, r in zip(abi["inputNames"], refs)],
                    outputs=[dict(name=n, tensor=r, dtype=str(v.numpy().dtype), shape=list(v.shape)) for n, r, v in zip(abi["outputNames"], outputs, flat)],
                    phase=phase, step=step, cpuMaxAbsErrors=abi["maxAbsErrors"]))
                self.files.add(filename)
                env[node], references[node] = value, tree_unflatten(outputs, structure)
            elif node.op == "output":
                output_nodes, _ = tree_flatten(node.args[0])
                outputs = [references[n] for n in output_nodes]
            else:
                raise RuntimeError(f"Unsupported orchestration node: {node.format_node()}")
        self.recipes[key] = dict(inputs=inputs, outputs=outputs, actions=self.actions[start:], outputExamples=examples)
        return outputs
