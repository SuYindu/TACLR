import os
from typing import Dict, Tuple
import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizer
from .utils_cls import create_label_tensor
from .utils_gen import create_prompt, pairs_to_string


def load_jsonl(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_json(file_path, lines=True)


def load_dataset(dir_path: str) -> Dict[str, pd.DataFrame]:
    train_file_path = os.path.join(dir_path, "train_large.jsonl")
    test_file_path = os.path.join(dir_path, "test.jsonl")
    return {
        'train': load_jsonl(train_file_path),
        'test': load_jsonl(test_file_path)
    }

def create_taxonomy(dataset_dict: Dict[str, pd.DataFrame]):
    taxonomy = {}

    for df in dataset_dict.values():
        for _, row in df.iterrows():
            category = row['category']
            if category not in taxonomy:
                taxonomy[category] = {}
                
            for attribute, value_scores in row['target_scores'].items():
                if attribute not in taxonomy[category]:
                    taxonomy[category][attribute] = set()
                
                for value in value_scores:
                    if value != "n/a":
                        taxonomy[category][attribute].add(value)
    
    return taxonomy


def preprocess(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    paradigm: str,
    label_to_idx: Dict[Tuple, int] = None,
    category_to_guidelines: Dict[str, str] = None,
    example_selector = None,
    do_train: bool = False
):
    processed_data = []

    for _, row in df.iterrows():
        sample = {
            'title': row['input_title'],
            'description': row['input_description'],
            'category': row['category'],
            'pairs': {
                attribute: [value for value in values if value != "n/a"]
                for attribute, values in row['target_scores'].items()
            }
        }

        if paradigm == 'classification':
            if label_to_idx is None:
                raise ValueError("Label mapping required for classification")

            text = f"title: {sample['title']}\ndescription: {sample['description']}"
            sample['labels'] = create_label_tensor(
                sample['category'],
                sample['pairs'],
                label_to_idx
            )
        else:  # generation
            if category_to_guidelines is None:
                raise ValueError("Category guidelines required for generation")
                
            guidelines = category_to_guidelines[sample['category']]
            if example_selector is not None:
                few_shot_examples = example_selector.select_examples(sample)
            else:
                few_shot_examples = None
            
            text = create_prompt(
                tokenizer,
                guidelines,
                sample['title'],
                sample['description'],
                few_shot_examples=few_shot_examples
            )
            
            if do_train:
                prompt_length = len(tokenizer(text)['input_ids'])
                target = pairs_to_string(sample['pairs'])
                text = text + target
                
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            return_tensors=None
        )
        
        sample.update({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
        })
        
        if paradigm == 'generation' and do_train:
            labels = [-100] * prompt_length + tokenized['input_ids'][prompt_length:]
            sample['labels'] = labels
            
        processed_data.append(sample)
    
    return Dataset.from_list(processed_data)
