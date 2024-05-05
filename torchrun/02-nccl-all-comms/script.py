# Derived from: https://github.com/pytorch/pytorch/blob/main/test/distributed/test_nccl.py

import os
import re
import sys

import torch
import torch.cuda
import torch.cuda.nccl as nccl
import torch.distributed as c10d
import torch.distributed as dist


def check_env_var_exists(var_name, dtype=str):
    return None if not var_name in os.environ else dtype(os.environ[var_name])


# set by torchrun
LOCAL_RANK: int = check_env_var_exists("LOCAL_RANK", int)
WORLD_SIZE: int = check_env_var_exists("WORLD_SIZE", int)

# set by jobset
WORLD_RANK: int = check_env_var_exists("RANK", int)
MASTER_ADDR: str = check_env_var_exists("MASTER_ADDR", str)


nGPUs = torch.cuda.device_count()


def test_broadcast(device, dtype):
    expected = torch.zeros(128).uniform_().to(dtype=dtype, device=device)
    tensors = [expected.cuda()]
    for device in range(1, torch.cuda.device_count()):
        tensors.append(torch.zeros(128, dtype=dtype, device=device))

    nccl.broadcast(tensors)
    for i in range(torch.cuda.device_count()):
        assert torch.allclose(tensors[i], expected), f"❌ Broadcast raw tensors failed on device {i}"
    print("✅ NCCL Broadcast raw tensors passed")

    # Test with tuple
    tensors = [expected.cuda()]
    for device in range(1, torch.cuda.device_count()):
        tensors.append(torch.zeros(128, dtype=dtype, device=device))

    nccl.broadcast(tuple(tensors))
    for i in range(torch.cuda.device_count()):
        assert torch.allclose(tensors[i], expected), f"❌ Broadcast tuple failed on device {i}"
    print("✅ NCCL Broadcast tuple passed")


def test_reduce(device, dtype):
    gpu_tensors = [
        torch.zeros(128).uniform_().to(dtype=dtype, device=device) for i in range(nGPUs)
    ]
    expected = torch.zeros(128, dtype=dtype, device=device)
    for t in gpu_tensors:
        expected.add_(t)

    tensors = [gpu_tensors[i].cuda(i) for i in range(nGPUs)]
    nccl.reduce(tensors)

    assert torch.allclose(tensors[0], expected), "❌ Reduce raw tensors failed"
    print("✅ NCCL Reduce raw tensors passed")

    # Test with tuple
    tensors = [gpu_tensors[i].cuda(i) for i in range(nGPUs)]
    nccl.reduce(tuple(tensors))

    assert torch.allclose(tensors[0], expected), "❌ Reduce tuple failed"
    print("✅ NCCL Reduce tuple passed")

def test_all_reduce(device, dtype):
    gpu_tensors = [
        torch.zeros(128).uniform_().to(dtype=dtype, device=device) for i in range(nGPUs)
    ]
    expected = torch.zeros(128, dtype=dtype, device=device)
    for t in gpu_tensors:
        expected.add_(t)

    tensors = [gpu_tensors[i].cuda(i) for i in range(nGPUs)]
    nccl.all_reduce(tensors)

    for tensor in tensors:
        assert torch.allclose(tensor, expected), "❌ All reduce raw tensors failed"
    print("✅ NCCL All reduce raw tensors passed")

    # Test with tuple.
    tensors = tuple(gpu_tensors[i].cuda(i) for i in range(nGPUs))
    nccl.all_reduce(tensors)

    for tensor in tensors:
        assert torch.allclose(tensor, expected), "❌ All reduce tuple failed"
    print("✅ NCCL All reduce tuple passed")

    # Test with set.
    tensors = {gpu_tensors[i].cuda(i) for i in range(nGPUs)}
    nccl.all_reduce(tensors)

    for tensor in tensors:
        assert torch.allclose(tensor, expected), "❌ All reduce set failed"
    print("✅ NCCL All reduce set passed")


def test_all_gather(device, dtype):
    gpu_inputs = [torch.zeros(128).uniform_().to(dtype=dtype, device=device) for i in range(nGPUs)]
    expected = torch.cat(gpu_inputs, 0)

    inputs = [gpu_inputs[i] for i in range(nGPUs)]
    outputs = [
        torch.zeros(128 * nGPUs, device=i, dtype=dtype) for i in range(nGPUs)
    ]
    nccl.all_gather(inputs, outputs)

    for tensor in outputs:
        assert torch.allclose(tensor, expected), "❌ All gather raw tensors failed"
    print("✅ NCCL All gather raw tensors passed")

    # Test with tuple.
    inputs = [gpu_inputs[i] for i in range(nGPUs)]
    outputs = [
        torch.zeros(128 * nGPUs, device=i, dtype=dtype) for i in range(nGPUs)
    ]
    nccl.all_gather(tuple(inputs), tuple(outputs))

    for tensor in outputs:
        assert torch.allclose(tensor, expected), "❌ All gather tuple failed"
    print("✅ NCCL All gather tuple passed")


def test_reduce_scatter(device, dtype):
    in_size = 32 * nGPUs
    out_size = 32

    gpu_inputs = [
        torch.zeros(in_size).uniform_().to(dtype=dtype, device=device) for i in range(nGPUs)
    ]
    expected = torch.zeros(in_size, dtype=dtype, device=device)
    for t in gpu_inputs:
        expected.add_(t)
    expected = expected.view(nGPUs, 32)

    inputs = [gpu_inputs[i].cuda(i) for i in range(nGPUs)]
    outputs = [torch.zeros(out_size, device=i, dtype=dtype) for i in range(nGPUs)]
    nccl.reduce_scatter(inputs, outputs)

    for i in range(nGPUs):
        assert torch.allclose(outputs[i], expected[i]), f"❌ Reduce scatter raw tensors failed on device {i}"
    print("✅ NCCL Reduce scatter raw tensors passed")

    # Test with tuple
    inputs = [gpu_inputs[i].cuda(i) for i in range(nGPUs)]
    outputs = [torch.zeros(out_size, device=i, dtype=dtype) for i in range(nGPUs)]
    nccl.reduce_scatter(tuple(inputs), tuple(outputs))

    for i in range(nGPUs):
        assert torch.allclose(outputs[i], expected[i]), f"❌ Reduce scatter tuple failed on device {i}"
    print("✅ NCCL Reduce scatter tuple passed")

if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.float32
    dist.init_process_group('nccl', rank=WORLD_RANK, world_size=WORLD_SIZE)
    test_broadcast(device, dtype)
    test_reduce(device, dtype)
    test_all_reduce(device, dtype)
    test_all_gather(device, dtype)
    test_reduce_scatter(device, dtype)
    dist.destroy_process_group()