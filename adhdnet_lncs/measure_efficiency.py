"""Measure parameters, FP32 weight storage, GFLOPs and inference latency.

Usage from the training notebook, after defining the two models:

    from measure_efficiency import report
    report(adhdnet, "ADHDNet")
    report(baseline, "Standard Conv3D")

Assumes model(mri, pheno) with mri of shape (B, 1, 64, 64, 64) and pheno of
shape (B, 5). Adjust make_inputs() if the forward signature differs.
FLOPs are counted with fvcore (pip install fvcore); fvcore counts one fused
multiply-add as one FLOP, so the reported number is GMACs-equivalent.
"""
import time

import torch


def make_inputs(device, batch_size=1):
    mri = torch.randn(batch_size, 1, 64, 64, 64, device=device)
    pheno = torch.randn(batch_size, 5, device=device)
    return mri, pheno


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_gflops(model, inputs):
    from fvcore.nn import FlopCountAnalysis

    flops = FlopCountAnalysis(model, inputs)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return flops.total() / 1e9


@torch.no_grad()
def time_inference(model, inputs, warmup=20, runs=100):
    device = inputs[0].device
    for _ in range(warmup):
        model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs * 1000  # ms per sample (batch 1)


def report(model, name, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    inputs = make_inputs(device)
    params = count_params(model)
    hw = torch.cuda.get_device_name() if device == "cuda" else "CPU"
    print(f"{name}")
    print(f"  Parameters          : {params:,} ({params / 1e6:.4f}M)")
    print(f"  FP32 weight storage : {params * 4 / 2**20:.2f} MiB")
    print(f"  GFLOPs (fvcore)     : {count_gflops(model, inputs):.3f}")
    print(f"  Inference time      : {time_inference(model, inputs):.2f} ms/sample on {hw}")
