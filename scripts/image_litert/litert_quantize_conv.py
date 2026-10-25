#!/usr/bin/env python3
"""LiteRT CLI entry point with a guarded ai-edge-quantizer 0.10.0 conv fix.

Its >32 MiB path reshapes 4D weights to 2D before broadcasting 4D scales.
Broadcast in the original shape first, then reshape both. This leaves installed
packages untouched and changes neither recipe nor quantization arithmetic.
"""
import importlib.metadata
import inspect


def install_fix():
    from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u
    assert importlib.metadata.version('ai-edge-quantizer') == '0.10.0'
    source = inspect.getsource(u.uniform_quantize)
    old = ('scales = np.broadcast_to(scales, tensor_data.shape)\n'
           '    zero_points = np.broadcast_to(zero_points, tensor_data.shape)')
    new = ('scales = np.broadcast_to(scales, output_shape).reshape(tensor_data.shape)\n'
           '    zero_points = np.broadcast_to(zero_points, output_shape).reshape(tensor_data.shape)')
    assert source.count(old) == 1, 'Quantizer implementation changed; re-audit the compatibility fix'
    exec(compile(source.replace(old, new), __file__, 'exec'), u.__dict__)


if __name__ == '__main__':
    install_fix()
    from litert_cli.litert import cli
    cli()
