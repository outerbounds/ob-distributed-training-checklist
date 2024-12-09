# Outerbounds multi-node workload checklist

This repository is for validating and benchmarking multi-node infrastructure managed by Outerbounds, and accessed via Metaflow decorators such as `@parallel`, `@torchrun`, `@deepspeed`, and `@metaflow_ray`. This checklist can help diagnose inter-node communication issues, verify a correct setup, and help ensure Outerbounds customers are seeing expected inter-node communcication performance across any supported compute platform.

The repository helps to:
- Validate NCCL operations across your distributed training cluster
- Benchmark communication performance between nodes
- Identify potential networking bottlenecks
- Verify cloud-specific configurations

Use the repository to:
- Decide whether you want to follow the `torchrun` track or the `deepspeed` track.
    - If you use one or the other framework for launching multi-node programs already, choose it. If you are not sure, choose `torchrun`.
- Run all NCCL operations.
- Run network performance tests. 

## Torchrun tests

Run these benchmarks to test all NCCL operations are working successfully using the `torchrun` launcher.

### All NCCL comms

Tests fundamental NCCL communication primitives (broadcast, reduce, scatter, gather) across multiple nodes to ensure basic distributed operations are functioning correctly. This test verifies that all nodes can communicate effectively using PyTorch's distributed communication backend. 

```bash
make torchrun-nccl-all-comms
```

### All reduce profile

Measures the performance of all-reduce operations, which are critical for gradient synchronization in distributed training. This benchmark provides detailed timing information and bandwidth measurements to help identify potential bottlenecks in your multi-node setup.

```bash
make torchrun-all-reduce-profile
```

### Run both back-to-back

Executes both the NCCL communications test and all-reduce profiling sequentially to get a comprehensive view of your cluster's communication capabilities.

```bash
make torchrun-all
```

## Deepspeed tests

A suite of tests specifically for DeepSpeed- and MPI-based distributed training configurations.
```bash
make deepspeed-all
```

### All reduce profile

Evaluates the performance of DeepSpeed's all-reduce operations, providing detailed metrics about communication efficiency and potential bottlenecks in gradient synchronization.

```bash
make deepspeed-all-reduce-profile
```

### All NCCL comms

Tests DeepSpeed's implementation of NCCL primitives to ensure reliable communication between nodes in your distributed training setup. This mirrors the torchrun example for NCCL comms.

```bash
make deepspeed-nccl-all-comms
```

### Deepspeed implementation communications benchmark

Deepspeed also implements its own interfaces to NCCL operations. This test runs them.
```bash
make deepspeed-comms-bench
```

## Cloud-specific tests

### Are all my pods being created in the same AZ?

Verifies that all compute nodes are being allocated within the same availability zone to minimize network latency and potential cross-AZ data transfer costs. This is crucial for optimal performance in distributed training workloads.

```bash
make aws-same-az
make azure-same-az
```