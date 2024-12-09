from metaflow import FlowSpec, step, torchrun, kubernetes, current
from metaflow.profilers import gpu_profile


# REPLACE WITH YOUR DESIRED IMAGE AND RESOURCE SETTINGS 
IMAGE = 'docker.io/eddieob/hello-torchrun:12' 
N_NODES = 2
N_GPU = 1


class TorchrunNCCLComms(FlowSpec):

    '''
    The benchmark performs each of the core NCCL operations.
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
    TorchrunNCCLComms()