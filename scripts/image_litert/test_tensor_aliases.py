"""Regression for token passthrough outputs versus distinct, same-shaped outputs."""
from pathlib import Path
import tempfile
import unittest

import litert_torch
import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter
from tensor_aliases import signature_aliases


class Passthrough(torch.nn.Module):
    def forward(self, tokens):
        return tokens, tokens + 1


class TensorAliasTest(unittest.TestCase):
    def test_real_int64_graph_preserves_alias_and_distinguishes_same_shape_output(self):
        tokens=torch.arange(4,dtype=torch.int64).reshape(1,4)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'alias.tflite'
            litert_torch.convert(Passthrough().eval(),(tokens,),enable_x64=True).export(str(path))
            self.assertEqual({'output_0':'args_0'},signature_aliases(path,'serving_default'))
            runner=Interpreter(model_path=str(path)).get_signature_runner('serving_default')
            result=runner(args_0=tokens.numpy())
            np.testing.assert_array_equal(result['output_0'],tokens.numpy())
            np.testing.assert_array_equal(result['output_1'],tokens.numpy()+1)


if __name__=='__main__': unittest.main()
