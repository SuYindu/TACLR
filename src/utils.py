import os
import yaml
import json
import random
import argparse
import numpy as np
import torch
from accelerate import Accelerator
from transformers import TrainingArguments


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_run_name(args, n_process):
    model_name_short = args.model_name.split('/')[-1].lower()
    
    if args.paradigm == "classification":
        return (
            f"{args.paradigm}"
            f"_{model_name_short}"
            f"_lr{args.learning_rate}"
            f"_bs{args.per_device_train_batch_size}*{n_process}"
            f"_ep{args.num_train_epochs}"
            f"_seed{args.seed}"
        )
    elif args.paradigm == "retrieval":
        return (
            f"{args.paradigm}"
            + f"_{model_name_short}"
            + (f"_dim{args.dim_proj}" if args.dim_proj else "")
            + f"_temp{args.temperature}"
            + f"_ns{args.num_samples}"
            + f"_lr{args.learning_rate}"
            + f"_bs{args.per_device_train_batch_size}*{n_process}"
            + (f"_ep{args.num_train_epochs}" if getattr(args, 'max_steps', None) is None else f"_st{args.max_steps}")
            + f"_seed{args.seed}"
        )
    else:  # generation
        if args.do_train:
            return (
                f"{args.paradigm}"
                f"_ft_{model_name_short}"
                f"_lr{args.learning_rate}"
                f"_bs{args.per_device_train_batch_size}*{args.gradient_accumulation_steps}*{n_process}"
                f"_ep{args.num_train_epochs}"
                f"_seed{args.seed}"
            )
        elif args.n_shots == 0:
            return (
                f"{args.paradigm}"
                f"_icl_{model_name_short}"
                f"_zero_shot"
                f"_bs{args.per_device_eval_batch_size}*{n_process}"
                f"_seed{args.seed}"
            )
        else:
            return (
                f"{args.paradigm}"
                f"_icl_{model_name_short}"
                f"_{args.n_shots}-shot"
                f"_{args.example_selector}"
                f"_bs{args.per_device_eval_batch_size}*{n_process}"
                f"_seed{args.seed}"
            )


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        specific_config = yaml.safe_load(f)
    if 'defaults' not in specific_config:
        return specific_config
        
    merged_config = {}
    config_dir = os.path.dirname(config_path)
    base_configs = specific_config.pop('defaults')
    for base_name in base_configs:
        base_path = os.path.join(config_dir, f"{base_name}.yaml")
        with open(base_path, 'r') as f:
            merged_config.update(yaml.safe_load(f))
    merged_config.update(specific_config)

    return merged_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='Path to config file')
    parser.add_argument("--run_name", type=str, help='Override auto-generated run name')
    parser.add_argument("--seed", type=int, default=42, help='Random seed')
    parser.add_argument("--learning_rate", type=float, help='Override learning rate')
    parser.add_argument("--per_device_train_batch_size", type=int, help='Override train batch size')
    parser.add_argument("--per_device_eval_batch_size", type=int, help='Override eval batch size')
    parser.add_argument("--num_train_epochs", type=int, help='Override number of epochs')
    args = parser.parse_args()
    
    config = load_config(args.config)
    for k, v in config.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    
    if args.run_name is None:
        args.run_name = get_run_name(args, Accelerator().num_processes)

    return args


def get_training_args(args):
    return TrainingArguments(
        output_dir=args.checkpoint_dir,
        report_to="wandb",
        log_level='info',
        full_determinism=True,
        bf16=args.bf16,
        fp16=args.fp16,
        ddp_find_unused_parameters=False,

        do_train=args.do_train,
        do_eval=args.do_eval,
        optim=args.optim,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,

        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=args.load_best_model_at_end,
        greater_is_better=args.greater_is_better,
    )


def save_results(args, trainer, metrics):
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    accelerator = Accelerator()
    rank_suffix = f"_rank{accelerator.process_index}" if accelerator.num_processes > 1 else ""
    
    if accelerator.is_main_process:
        metrics_file = os.path.join(run_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    results_file = os.path.join(run_dir, f"results{rank_suffix}.txt")
    with open(results_file, 'w') as f:
        f.write('\n'.join(trainer.result_lines))
            
    if hasattr(trainer, 'conversations'):
        conversations_file = os.path.join(run_dir, f"conversations{rank_suffix}.txt")
        with open(conversations_file, 'w') as f:
            f.write('\n'.join(trainer.conversations))
