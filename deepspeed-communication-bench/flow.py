from metaflow import FlowSpec, step, deepspeed, kubernetes, current


IMAGE = "docker.io/eddieob/deepspeed:6"
N_NODES = 2
N_GPU = 1


class DeepspeedCommunicationBenchmark(FlowSpec):

    @step
    def start(self):
        self.next(self.torch_multinode, num_parallel=N_NODES)

    @kubernetes(image=IMAGE, gpu=N_GPU, shared_memory=12000)
    @deepspeed
    @step
    def torch_multinode(self):
        current.deepspeed.run(
            entrypoint="run_all.py",
            entrypoint_args={
                "all-reduce": "",
                "all-to-all": "",
                "broadcast": "",
                "scan": "",
                "pt2pt": "",
            }
        )
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    DeepspeedCommunicationBenchmark()