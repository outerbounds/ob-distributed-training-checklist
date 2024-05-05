from metaflow import FlowSpec, step, parallel, kubernetes, pypi, pypi_base

N_NODES = 4
GPU = 1
CPU = 1

@pypi_base(python='3.12')
class TestCheckAZs(FlowSpec):

    checks = [('az', 'Availability zones')]
    
    @step
    def start(self):
        self.next(self.distributed_step, num_parallel=N_NODES)

    @pypi(packages={'ec2-metadata': '2.13.0'})
    @kubernetes(gpu=GPU, cpu=CPU) # SET THIS TO YOUR DESIRED COMPUTE CONFIG
    @parallel
    @step
    def distributed_step(self):
        from ec2_metadata import ec2_metadata
        self.region = ec2_metadata.region
        self.az = ec2_metadata.availability_zone
        self.next(self.join)

    @step
    def join(self, inputs):
        self.data = {"region": [], "az": []}
        for i in inputs:
            self.data["region"].append(i.region)
            self.data["az"].append(i.az)
        self.next(self.end)

    @pypi(packages={'pandas': '2.2.2'})
    @step
    def end(self):
        import pandas as pd
        df = pd.DataFrame(self.data)
        for check_col, desc in self.checks:
            assert df[check_col].nunique() == 1, f"❌ {desc} are not the same ~ {df[check_col].unique()}"
            print(f"✅ {desc} are all in '{df[check_col].unique()[0]}'")

if __name__ == '__main__':
    TestCheckAZs()