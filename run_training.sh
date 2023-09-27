torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=101 --rdzv_endpoint="localhost:5972" main_training.py --model gpt2 --profile --profile_folder=trace/base --batch_size_training 4
