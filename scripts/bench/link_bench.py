#!/usr/bin/env python3
"""Measure the GPU-GPU link: host<->device and peer copies, and NCCL all-reduce latency/bandwidth.

Run inside the SGLang image with both GPUs visible:
  docker run --rm --gpus all --ipc host -v $PWD:/w lmsysorg/sglang:v0.5.21-cu130 \
      torchrun --nproc-per-node 2 /w/link_bench.py --out /w/link.json
Rank 0 also runs the copy tests before the all-reduce sweep. Peer copies are checked for correctness, not
just timed: on this AMD B550 host with the IOMMU in translated mode, CUDA reports peer access as available
but every peer write is dropped (AMD-Vi IO_PAGE_FAULT) and the destination stays zero. Run with
NCCL_P2P_DISABLE=1 there, or NCCL hangs waiting for data that never arrives.
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def timed(fn, iters):
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def copy_tests():
    res = {"peer_access": {f"{a}->{b}": torch.cuda.can_device_access_peer(a, b) for a in range(2) for b in range(2) if a != b}}
    n = 1 << 28  # 256 MiB
    for d in range(2):
        host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
        dev = torch.empty(n, dtype=torch.uint8, device=d)
        with torch.cuda.device(d):  # timed() synchronizes the current device only
            res[f"h2d_gpu{d}_GiBps"] = n / timed(lambda: dev.copy_(host, non_blocking=True), 10) / 2**30
            res[f"d2h_gpu{d}_GiBps"] = n / timed(lambda: host.copy_(dev, non_blocking=True), 10) / 2**30
    for src, dst in [(0, 1), (1, 0)]:
        x = torch.arange(1, 1025, dtype=torch.int32, device=src)
        y = torch.zeros(1024, dtype=torch.int32, device=dst)
        y.copy_(x)
        torch.cuda.synchronize(src), torch.cuda.synchronize(dst)
        res[f"p2p_{src}to{dst}_correct"] = bool((y.cpu() == x.cpu()).all())
    if not all(res[k] for k in res if k.endswith("_correct")):
        return res  # timing a copy that delivers nothing is meaningless
    a, b = torch.empty(n, dtype=torch.uint8, device=0), torch.empty(n, dtype=torch.uint8, device=1)
    res["p2p_0to1_GiBps"] = n / timed(lambda: b.copy_(a, non_blocking=True), 10) / 2**30
    res["p2p_1to0_GiBps"] = n / timed(lambda: a.copy_(b, non_blocking=True), 10) / 2**30
    sa, sb = torch.empty(4096, dtype=torch.uint8, device=0), torch.empty(4096, dtype=torch.uint8, device=1)
    res["p2p_4KiB_latency_us"] = timed(lambda: sb.copy_(sa, non_blocking=True), 2000) * 1e6
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    res = copy_tests() if rank == 0 else {}
    dist.init_process_group("nccl")
    dist.barrier()
    sweep = []
    # 10 KiB ~ one decode token's hidden state (5120 x bf16); 80 MiB ~ an 8,192-token prefill chunk
    for nbytes in [10 * 1024, 40 * 1024, 160 * 1024, 1 << 20, 8 << 20, 80 << 20]:
        x = torch.ones(nbytes // 2, dtype=torch.bfloat16, device=rank)
        iters = 200 if nbytes < (1 << 20) else 20
        dist.barrier()
        t = timed(lambda: dist.all_reduce(x), iters)
        sweep.append({"bytes": nbytes, "latency_us": t * 1e6, "busbw_GiBps": nbytes / t / 2**30})  # 2 ranks: busbw = algbw
    if rank == 0:
        res["nccl_version"] = ".".join(map(str, torch.cuda.nccl.version()))
        res["allreduce"] = sweep
        print(json.dumps(res, indent=1))
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
