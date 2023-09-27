import time
import tqdm
import torch
from dataclasses import dataclass

from torch.distributed.fsdp import StateDictType
from torch.utils.data import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import datasets_grammar as dg
from .base_config import base_config, fsdp_checkpointing_base, get_policy_base


@dataclass
class train_config(base_config):
    model_name = "gpt2"

    # important - if you want trackable loss stats, please ensure you use real data:
    use_real_data = False
    use_synthetic_data: bool = False

    use_deferred_init: bool = False

    # DDP
    use_ddp: bool = False
    ddp_bucket_size: float = 25
    ddp_use_gradient_view: bool = False

    # activation checkpointing
    fsdp_activation_checkpointing: bool = False
    hf_t5_checkpointing: bool = False

    # torch.compile
    use_torch_compile: bool = False

    # checkpoint models
    save_model_checkpoint: bool = False
    load_model_checkpoint: bool = False
    checkpoint_type = StateDictType.SHARDED_STATE_DICT
    dist_checkpoint_root_folder = "distributed_checkpoints"
    dist_checkpoint_folder = "gpt2_local_checkpoint"
    model_save_name = "gpt2-"
    checkpoint_folder = "training_checkpoints"
    checkpoint_max_save_count: int = (
        2  # number of 'best' checkpoints to save based on val loss
    )

    # optimizers load and save
    save_optimizer: bool = False
    load_optimizer: bool = False

    optimizer_checkpoint_file: str = "Adam-gpt2--1.pt"

    checkpoint_model_filename: str = "gpt2--1.pt"

    # datasets
    dataset_train = "datasets_grammar/grammar_train.csv"  # grammar_13k.csv
    dataset_test = "datasets_grammar/grammar_validation.csv"


def build_model(
    model_name: str,
    use_parallel_attention=False,
    use_fused_attention=False,
):
    cfg = train_config()
    if cfg.use_real_data:
        return GPT2LMHeadModel.from_pretrained(model_name)
    if model_name == "gpt2":
        configure = GPT2Config(
            n_embd=3072,
            n_head=48,
            n_layer=5,
            vocab_size=50257,
        )
    else:
        raise ValueError(f"model_name {model_name} not supported")
    return GPT2LMHeadModel(configure)


class GeneratedDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        super()
        self._input_shape = kwargs.get("input_shape", (1024,))
        self._input_type = kwargs.get("input_type", torch.long)
        self._len = kwargs.get("len", 1000000)
        self._num_classes = kwargs.get("num_classes", 32000)

    def __len__(self):
        return self._len

    def __getitem__(self, index: int):
        return {
            "source_ids": torch.randint(
                1, self._num_classes, self._input_shape, dtype=self._input_type
            ),
            "source_mask": torch.randint(
                1, self._num_classes, self._input_shape, dtype=self._input_type
            ),
            "target_ids": torch.randint(
                1, self._num_classes, self._input_shape, dtype=self._input_type
            ),
            "target_mask": torch.randint(
                1, self._num_classes, self._input_shape, dtype=self._input_type
            ),
        }


def get_dataset():
    cfg = train_config()
    if cfg.use_real_data:
        train_name = cfg.dataset_train
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        train_dataset = dg.get_dataset(tokenizer, train_name, 1024, 1024, True)
        return train_dataset
    return GeneratedDataset()


def get_policy():
    return get_policy_base({GPT2Block})


def fsdp_checkpointing(model):
    return fsdp_checkpointing_base(model, GPT2Block)


def train(
    model,
    data_loader,
    torch_profiler,
    optimizer,
    memmax,
    local_rank,
    tracking_duration,
    total_steps_to_run,
    use_parallel_attention=False,
    use_fused_attention=True,
    use_synthetic_data: bool = False,
    use_label_singular: bool = False,
    stats=None,
    lr_scheduler=None,
):
    cfg = train_config()
    model.train()
    if local_rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(data_loader)), colour="blue", desc="r0 Training Epoch"
        )
    batch_index = 0
    t0 = time.perf_counter()
    for batch in data_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        if optimizer:
            optimizer.zero_grad()
        output = model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["target_ids"],
        )
        loss = output["loss"]
        loss.backward()
        if optimizer:
            optimizer.step()

        if local_rank == 0:
            inner_pbar.update(1)
        if torch_profiler:
            torch_profiler.step()
        batch_index += 1
        mini_batch_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        if local_rank == 0:
            tracking_duration.append(mini_batch_time)
            if memmax:
                memmax.update()
        if (
            batch_index % cfg.log_every == 0
            and torch.distributed.get_rank() == 0
            and batch_index > 1
        ):
            print(
                f"step: {batch_index-1}: time taken for the last {cfg.log_every} steps is {mini_batch_time}, loss is {loss}"
            )

        if batch_index > total_steps_to_run:
            break
    if local_rank == 0:
        inner_pbar.close()
        print("tracking_duration", tracking_duration)
