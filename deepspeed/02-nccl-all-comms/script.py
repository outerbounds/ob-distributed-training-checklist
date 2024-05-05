# Derived from: https://github.com/pytorch/pytorch/blob/main/test/distributed/test_nccl.py

import os
import re
import sys

import torch
import deepspeed
import deepspeed.comm as dist

def check_env_var_exists(var_name, dtype=str):
    return None if not var_name in os.environ else dtype(os.environ[var_name])

deepspeed.init_distributed(dist_backend='nccl')

# set by torchrun
LOCAL_RANK: int = check_env_var_exists("LOCAL_RANK", int)
WORLD_SIZE: int = dist.get_world_size()

# set by jobset
WORLD_RANK: int = dist.get_rank()
MASTER_ADDR: str = check_env_var_exists("MASTER_ADDR", str)

def test_broadcast(device, dtype):
    # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#broadcast
    tensor = torch.tensor([0.]).float().cuda()
    if WORLD_RANK == 0:
        tensor = torch.tensor([42.]).float().cuda()
    dist.broadcast(tensor=tensor, src=0)
    assert torch.allclose(tensor, torch.tensor([42.]).float().cuda()), "❌ broadcast failed on {}".format(WORLD_RANK) 
    print("✅ NCCL broadcast passed on {}".format(WORLD_RANK))

def test_reduce(device, dtype):
    # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reduce
    tensor = torch.tensor([WORLD_RANK + 1.0]).float().cuda()
    expected_sum = torch.tensor(sum(range(1, WORLD_SIZE + 1))).float().cuda()
    dist.reduce(tensor=tensor, dst=0, op=dist.ReduceOp.SUM)
    if WORLD_RANK == 0:
        assert torch.allclose(tensor, expected_sum), "❌ reduce failed"
        print("✅ NCCL reduce passed")

def test_all_reduce(device, dtype):
    # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce
    tensor = torch.tensor([WORLD_RANK + 1.0]).float().cuda()
    expected_sum = torch.tensor(sum(range(1, WORLD_SIZE + 1))).float().cuda()
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    assert torch.allclose(tensor, expected_sum), "❌ all reduce failed on {}".format(WORLD_RANK)
    print("✅ NCCL all reduce passed on {}".format(WORLD_RANK))

def test_all_gather(device, dtype):
    # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather
    tensors_gather_storage_ls = [torch.tensor([0.0]).float().cuda() for _ in range(WORLD_SIZE)]
    my_rank_tensor = torch.tensor([WORLD_RANK + 1.0]).float().cuda()
    expected_ls = [torch.tensor([i + 1.0]).float().cuda() for i in range(WORLD_SIZE)]
    dist.all_gather(tensors_gather_storage_ls, my_rank_tensor)
    for i in range(WORLD_SIZE):
        assert torch.allclose(tensors_gather_storage_ls[i], expected_ls[i]), "❌ all gather failed on {}".format(WORLD_RANK)
    print("✅ NCCL all gather passed on {}".format(WORLD_RANK))

def test_reduce_scatter(device, dtype):
    # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reducescatter
    out_tensor = torch.tensor([0.0]).float().cuda()
    input_tensor_ls = [torch.tensor([21.0]).float().cuda(), torch.tensor([21.0]).float().cuda()]
    expected = torch.tensor([42.0]).float().cuda()
    dist.reduce_scatter(out_tensor, input_tensor_ls, op=dist.ReduceOp.SUM)
    assert torch.allclose(out_tensor, expected), "❌ reduce scatter failed on {}".format(WORLD_RANK)
    print("✅ NCCL reduce scatter passed on {}".format(WORLD_RANK))


if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.float32
    test_broadcast(device, dtype)
    test_reduce(device, dtype)
    test_all_reduce(device, dtype)
    test_all_gather(device, dtype)
    test_reduce_scatter(device, dtype)