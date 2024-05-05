from metaflow import FlowSpec, step, deepspeed, kubernetes, current


# REPLACE WITH YOUR DESIRED IMAGE AND RESOURCE SETTINGS 
IMAGE = "docker.io/eddieob/deepspeed:6" 
N_NODES = 2
N_GPU = 1


class DeepspeedAllreduce(FlowSpec):

    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=N_NODES)

    @kubernetes(image=IMAGE, gpu=N_GPU)
    @deepspeed
    @step
    def torch_multinode(self):
        current.deepspeed.run(entrypoint="script.py")
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    DeepspeedAllreduce()