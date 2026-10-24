#!/usr/bin/env python3
"""CPU smoke of the exact community FLUX graph ABI and reconstructed Android host loop.

Uses the same java.util.Random Gaussian starting noise as Android, with explicit
argument names and one graph resident at a time. This does not validate GPU/NPU.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.options import CpuOptions, Options
from PIL import Image
from transformers import AutoTokenizer


def run(path, *inputs):
    start = time.monotonic()
    model = CompiledModel.from_file(str(path), options=Options(cpu_options=CpuOptions(num_threads=8)))
    sig = next(iter(model.get_signature_list()))
    details = model.get_input_tensor_details(sig)
    ins, outs = {}, {}
    try:
        assert len(details) == len(inputs)
        for i, array in enumerate(inputs):
            name = f'args_{i}'
            assert list(array.shape) == list(details[name]['shape']), (path.name, name, array.shape, details[name]['shape'])
            ins[name] = model.create_input_buffer_by_name(sig, name)
            ins[name].write(np.ascontiguousarray(array, dtype=details[name]['dtype']))
        output = model.get_output_tensor_details(sig)
        for name in output:
            outs[name] = model.create_output_buffer_by_name(sig, name)
        model.run_by_name(sig, ins, outs)
        result = []
        for i in range(len(outs)):
            name = f'output_{i}'; d = output[name]
            a = outs[name].read(int(np.prod(d['shape'])), np.dtype(d['dtype'])).reshape(d['shape']).copy()
            assert np.isfinite(a).all(), (path, name, 'non-finite')
            result.append(a)
        print(path.name, round(time.monotonic() - start, 2), [a.shape for a in result], flush=True)
        return result
    finally:
        del ins, outs, model
        gc.collect()


def generate(model_dir, source, host_path, tokenizer_dir, noise_path, out):
    d = Path(model_dir); host = json.loads(Path(host_path).read_text())
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    prompt = 'a red apple on a wooden table, studio lighting'
    text = tok.apply_chat_template([dict(role='user', content=prompt)], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, padding='max_length', max_length=512, truncation=True, return_tensors='np')
    ids = enc['input_ids']
    embeds = np.memmap(d/'tokenizer/qwen_embed_fp16.bin', dtype='<f2', mode='r', shape=(151936, 2560))
    hidden = embeds[ids].astype(np.float32)
    real = int(enc['attention_mask'].sum())
    mask = np.zeros((512,512),np.float32)
    mask[np.triu_indices(512,1)] = -np.finfo(np.float32).max
    mask[:,real:] = -np.finfo(np.float32).max
    mask = np.tile(mask, (1,32,1,1))
    inv = 1 / 1_000_000 ** (np.arange(0,128,2,dtype=np.float32) / 128)
    angle = np.arange(512,dtype=np.float32)[:,None] * inv[None,:]
    angle = np.concatenate([angle,angle],-1)[None]
    ec, es = np.cos(angle), np.sin(angle)
    taps=[]
    for i in range(3):
        hidden=run(d/f'ke_enc{i}.tflite',hidden,mask,ec,es)[0]; taps.append(hidden)
    context=np.stack(taps,1).transpose(0,2,1,3).reshape(1,512,7680)
    ids=np.zeros((768,4),np.float64); ids[:512,3]=np.arange(512)
    ids[512:,1]=np.repeat(np.arange(16),16); ids[512:,2]=np.tile(np.arange(16),16)
    angle=np.concatenate([ids[:,i:i+1]/2000**(np.arange(16)[None,:]/16) for i in range(4)],-1)
    c=np.cos(angle).astype(np.float32)[None,:,None,:]; s=np.sin(angle).astype(np.float32)[None,:,None,:]
    lat=np.fromfile(noise_path,dtype='<f4').reshape(1,256,128)
    sigmas=np.array(host['sigmas'],np.float32)
    for k in range(4):
        temb=np.array(host['temb'][k:k+1],np.float32)
        im,tx,mi,mt,ms=run(d/'kc_prep.tflite',lat,context,temb)
        for i in range(2): im,tx=run(d/f'kc_double{i}.tflite',im,tx,c,s,mi,mt)
        joint=np.concatenate([tx,im],1)
        for i in range(4): joint=run(d/f'kc_single{i}.tflite',joint,c,s,ms)[0]
        v=run(d/'kc_final.tflite',joint,temb)[0]
        lat += (sigmas[k+1]-sigmas[k])*v
    z=lat[0]*np.array(host['bnScale'],np.float32)+np.array(host['bnShift'],np.float32)
    z=z.reshape(16,16,32,2,2).transpose(2,0,3,1,4).reshape(1,32,32,32)
    rgb=run(d/'kv_vae.tflite',z)[0][0]
    assert float(rgb.std()) > .01
    Image.fromarray(np.clip((rgb.transpose(1,2,0)/2+.5)*255+.5,0,255).astype(np.uint8)).save(out)
    print('SAVED',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['model-dir','source','host-path','tokenizer-dir','noise-path','out']: p.add_argument('--'+key,required=True)
    a=p.parse_args(); generate(**vars(a))
