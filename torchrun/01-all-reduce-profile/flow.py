from metaflow import FlowSpec, step, torchrun, kubernetes, current
from metaflow.profilers import gpu_profile


# REPLACE WITH YOUR DESIRED IMAGE AND RESOURCE SETTINGS 
IMAGE = 'docker.io/eddieob/hello-torchrun:12' 
N_NODES = 2
N_GPU = 1


class TorchrunAllreduce(FlowSpec):

    '''
    The benchmark performs the following:
        - Initializes a distributed process group using the specified backend (default: NCCL)
        - Creates a large tensor (1000³ elements, ~4GB with float32) on each GPU
        - Verifies correctness of all-reduce operation
        - Measures performance over multiple trials
        - Reports throughput and bus bandwidth metrics
    '''

    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=N_NODES)

    @gpu_profile(interval=1)
    @kubernetes(image=IMAGE, gpu=N_GPU)
    @torchrun
    @step
    def torch_multinode(self):
        current.torch.run(entrypoint="script.py")
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    TorchrunAllreduce()