from metaflow import FlowSpec, step, parallel, kubernetes, pypi, pypi_base

N_NODES = 4
GPU = 1
CPU = 1

@pypi_base(python='3.12')
class TestCheckAZs(FlowSpec):

    # checks = [('az', 'Availability zones'), ('region', 'Regions'), ('placement_group', 'Placement groups')]
    checks = [('az', 'Availability zones'), ('region', 'Regions')]
    
    @step
    def start(self):
        self.next(self.distributed_step, num_parallel=N_NODES)

    @kubernetes(gpu=GPU, cpu=CPU) # SET THIS TO YOUR DESIRED COMPUTE CONFIG
    @parallel
    @step
    def distributed_step(self):
        import requests
        import json
        # https://learn.microsoft.com/en-us/azure/virtual-machines/instance-metadata-service?tabs=linux
        url = "http://169.254.169.254/metadata/instance/compute?api-version=2023-07-01&format=json"
        headers = {'Metadata': 'true'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        vm_metadata = response.json()
        self.region = vm_metadata['location']
        self.az = vm_metadata['zone'] 
        # AZ is log is empty. why? 
        # placement group is narrower, more ideal for dist training, anyway.
        self.placement_group = vm_metadata['placementGroupId']
        self.next(self.join)

    @step
    def join(self, inputs):
        self.data = {"region": [], "az": [], "placement_group": []}
        for i in inputs:
            self.data["region"].append(i.region)
            self.data["az"].append(i.az)
            self.data["placement_group"].append(i.placement_group)
        self.next(self.end)

    @pypi(packages={'pandas': '2.2.2'})
    @step
    def end(self):
        import pandas as pd
        df = pd.DataFrame(self.data)
        for check_col, desc in self.checks:
            assert df[check_col].nunique() == 1, f"❌ {desc} are not the same ~ {df[check_col].unique()}"
            print(f"✅ {desc} are all in '{df[check_col].unique()[0]}'")
        # assert df['placement_group'].nunique() == 1, f"Placement groups are not the same ~ {df['placement_group'].unique()}"
        # print(f"✅ Placement groups are all in {df['placement_group'].unique()[0]}")

if __name__ == '__main__':
    TestCheckAZs()