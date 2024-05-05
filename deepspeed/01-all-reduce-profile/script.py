import os
import socket
import argparse
import torch
# import torch.distributed as dist

import deepspeed
import deepspeed.comm as dist


def check_env_var_exists(var_name, dtype=str):
    return None if not var_name in os.environ else dtype(os.environ[var_name])


deepspeed.init_distributed(dist_backend='nccl')


# set by torchrun
LOCAL_RANK: int = check_env_var_exists("LOCAL_RANK", int)
WORLD_SIZE: int = check_env_var_exists("WORLD_SIZE", int)

# set by jobset
WORLD_RANK: int = check_env_var_exists("RANK", int)
MASTER_ADDR: str = check_env_var_exists("MASTER_ADDR", str)


def get_bw(size, duration, bw_unit='Gbps'):
    # https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/benchmarks/communication/utils.py#L107C1-L132C23
    n = dist.get_world_size()
    tput = (size * 2 / duration)
    busbw = (size / duration) * (2 * (n - 1) / n)
    if bw_unit == 'Gbps':
        tput *= 8
        busbw *= 8
    return tput, busbw

def get_metric_strings(tput, busbw, duration, print_without_units=False):
    duration_ms = duration * 1e3
    duration_us = duration * 1e6
    tput = f'{tput / 1e9:.3f}'
    busbw = f'{busbw /1e9:.3f}'

    if duration_us < 1e3 or print_without_units:
        duration = f'{duration_us:.3f}'
        if not print_without_units:
            duration += ' us'
    else:
        duration = f'{duration_ms:.3f} ms'
    return tput, busbw, duration

def run(n_trials=50):
    N, M, K = 1000, 1000, 1000 # 1000^3 is ~ 4GB with float32
    tensor = torch.ones((N, M, K), dtype=torch.float32).cuda() * (1 + WORLD_RANK)

    # run once to verify correctness
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected_sum = torch.tensor(sum(range(1, 1 + WORLD_SIZE)), dtype=tensor.dtype, device=tensor.device)
    assert torch.allclose(tensor, expected_sum, atol=1e-5), "❌ All reduce failed"
    print("✅ All reduce passed")

    # run multiple trials to measure network performance
    size = tensor.nelement() * tensor.element_size()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_trials):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    end_event.record()
    torch.cuda.synchronize()

    # metrics on the wire
    duration = start_event.elapsed_time(end_event) / 1000
    avg_duration = duration / n_trials
    tput, busbw = get_bw(size, avg_duration)
    tput_str, busbw_str, duration_str = get_metric_strings(tput, busbw, avg_duration)
    if WORLD_RANK == 0:
        desc = f"all_reduce({N}x{M}x{K}) ~ est. {size / 1e9:.3f} GB"
        header = f"\n---- Performance of all reduce on {WORLD_SIZE} devices ---------------------------------------------------------\n"
        header += f"{'Size (Bytes)':20s} {'Description':25s} {'Duration':20s} {'Throughput (Gbps)':20s} {'BusBW (Gbps)':20s}\n"
        header += "----------------------------------------------------------------------------------------------------"
        print(header)
        print(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")

if __name__ == "__main__":
    run()