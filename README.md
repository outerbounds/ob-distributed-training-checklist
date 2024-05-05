This document is a recipe to help you verify your distributed training cluster managed by the Outerbounds.

### Do everything
```bash
make all-aws
make all-azure
```

### Are all my pods being created in the same AZ?
```bash
make aws-same-az
make azure-same-az
```

### Torchrun
```bash
make torchrun-all
```

#### All reduce profile
```bash
make torchrun-all-reduce-profile
```

#### All NCCL comms
```bash
make torchrun-nccl-all-comms
```

### Deepspeed
```bash
make deepspeed-all
```

#### All reduce profile
```bash
make deepspeed-all-reduce-profile
```

#### All NCCL comms
```bash
make deepspeed-nccl-all-comms
```

#### Multi-node communications benchmark
```bash
make deepspeed-comms-bench
```