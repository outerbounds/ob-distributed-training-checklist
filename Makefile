aws-same-az:
	python 00-all-pods-same-az-aws.py --environment=pypi run

azure-same-az: 
	python 00-all-pods-same-az-azure.py --environment=pypi run

torchrun-all-reduce-profile:
	cd torchrun/01-all-reduce-profile && python flow.py run

torchrun-nccl-all-comms:
	cd torchrun/02-nccl-all-comms && python flow.py run

torchrun-all: torchrun-all-reduce-profile torchrun-nccl-all-comms

deepspeed-all-reduce-profile:
	cd deepspeed/01-all-reduce-profile && python flow.py run

deepspeed-nccl-all-comms:
	cd deepspeed/02-nccl-all-comms && python flow.py run

deepspeed-comms-bench:
	cd deepspeed-communication-bench && python flow.py run

deepspeed-all: deepspeed-all-reduce-profile deepspeed-nccl-all-comms deepspeed-comms-bench

all-aws: aws-same-az torchrun-all deepspeed-all

all-azure: azure-same-az torchrun-all deepspeed-all