import os
from transformers import AutoTokenizer
from data import (
    load_dataset, create_taxonomy, preprocess, BaseDataCollator,
    create_label_mapping, ClassificationDataCollator, 
    create_category_guidelines, GenerationDataCollator, FewShotExampleSelector
)
from model import PlmForClassification, LlmForGeneration, PlmForRetrieval
from trainer import ClassificationTrainer, GenerationTrainer, RetrievalTrainer
from utils import set_seed, get_args, get_training_args, save_results


def main(args):
    set_seed(args.seed)

    dataset_dict = load_dataset(args.data_dir)
    taxonomy = create_taxonomy(dataset_dict)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left" if args.paradigm == "generation" else "right",
        token=os.environ.get("HF_TOKEN"),
    )
    
    if args.paradigm == "classification":
        idx_to_label, label_to_idx = create_label_mapping(taxonomy)
        train_dataset = preprocess(
            dataset_dict['train'], 
            tokenizer, 
            args.paradigm, 
            label_to_idx=label_to_idx, 
        )
        eval_dataset = preprocess(
            dataset_dict['test'], 
            tokenizer, 
            args.paradigm, 
            label_to_idx=label_to_idx
        )
        data_collator = ClassificationDataCollator(tokenizer)
        model = PlmForClassification(args.model_name, idx_to_label, label_to_idx)
        trainer_cls = ClassificationTrainer
        trainer_kwargs = {
            "idx_to_label": idx_to_label,
            "threshold": args.threshold
        }
    elif args.paradigm == "retrieval":
        train_dataset = preprocess(
            dataset_dict['train'], 
            tokenizer, 
            args.paradigm, 
            do_train=args.do_train
        )
        eval_dataset = preprocess(
            dataset_dict['test'], 
            tokenizer, 
            args.paradigm
        )
        data_collator = BaseDataCollator(tokenizer)
        model = PlmForRetrieval(
            model_name=args.model_name,
            taxonomy=taxonomy,
            dim_proj=args.dim_proj,
            temperature=args.temperature,
            num_samples=args.num_samples
        )
        trainer_cls = RetrievalTrainer
        trainer_kwargs = {}
    else:  # generation
        tokenizer.pad_token = tokenizer.eos_token
        category_guidelines = create_category_guidelines(taxonomy)
        if not args.do_train and args.n_shots > 0:
            example_selector = FewShotExampleSelector(
                dataset_dict['train'],
                n_shots=args.n_shots,
                selector_type=args.example_selector
            )
        else:
            example_selector = None
        train_dataset = preprocess(
            dataset_dict['train'], 
            tokenizer, 
            args.paradigm, 
            category_to_guidelines=category_guidelines,
            example_selector=example_selector,
            do_train=args.do_train
        )
        eval_dataset = preprocess(
            dataset_dict['test'], 
            tokenizer, 
            args.paradigm, 
            category_to_guidelines=category_guidelines,
            example_selector=example_selector
        )
        data_collator = GenerationDataCollator(tokenizer)
        model = LlmForGeneration(tokenizer, args.model_name, args)
        trainer_cls = GenerationTrainer
        trainer_kwargs = {"processor": tokenizer}

    training_args = get_training_args(args)
    common_trainer_kwargs = {
        "taxonomy": taxonomy,
        "args": training_args,
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }    
    trainer_kwargs.update(common_trainer_kwargs)
    
    trainer = trainer_cls(**trainer_kwargs)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        metrics = trainer.evaluate()
        save_results(args, trainer, metrics)


if __name__ == "__main__":
    args = get_args()
    main(args)
