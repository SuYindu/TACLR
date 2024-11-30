import json
import random
import pandas as pd
import torch
from .wdc_collator import BaseDataCollator


class GenerationDataCollator(BaseDataCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        
        if any('labels' in f for f in features):
            labels = torch.full_like(batch['input_ids'], -100)
            labels[:, :-1] = batch['input_ids'][:, 1:]
            batch['labels'] = labels

        return batch


def get_normalization_guidelines_from_csv(category):
    MEASUREMENT_CATEGORIES = ["Home And Garden", "Office Products"]
    MEASUREMENT_ATTRIBUTES = ['Width', 'Height', 'Depth', 'Length']
    SIZE_NORMALIZATION_GUIDE = """
        Convert and standardize the Width/Height/Depth/Length measurements to centimeters (cm). 
        If the Width/Height/Depth/Length is not already in cm, convert it from inches, mm, meters, yards, or feet to cm.
        If no unit is specified, assume the measurement is in inches.
        For Width/Height/Depth/Length ranges, use the larger value in the range for conversion.
        Output the final Width/Height/Depth/Length value as a single numeric figure in cm, rounded to one decimal place.
        Exclude any unit indicators in the output.
    """

    descriptions_path = '../data/wdc_normalized/descriptions.csv'
    descriptions_csv = pd.read_csv(descriptions_path, sep=';')
    descriptions_csv = descriptions_csv[['Category', 'Attribute', 'Normalization_params', 'Normalization_instruction']]
    descriptions_csv["Normalization_params"] = descriptions_csv["Normalization_params"].str.strip("[]").str.replace("'", "")

    if category in MEASUREMENT_CATEGORIES:
        measurement_rule = pd.DataFrame([{
            'Category': category,
            'Attribute': 'Width/Height/Depth/Length',
            'Normalization_params': 'Unit Conversion',
            'Normalization_instruction': SIZE_NORMALIZATION_GUIDE
        }])
        descriptions_csv = pd.concat([descriptions_csv, measurement_rule], ignore_index=True)
        # Remove individual measurement attributes if they exist
        descriptions_csv = descriptions_csv[~descriptions_csv['Attribute'].isin(MEASUREMENT_ATTRIBUTES)]

    # Filter guidelines for the category
    guidelines = descriptions_csv[descriptions_csv["Category"] == category]
    
    # Build guidelines string
    guidelines_parts = []
    attributes_without_instructions = []

    for _, row in guidelines.iterrows():
        if pd.isna(row["Normalization_instruction"]):
            attributes_without_instructions.append(row["Attribute"])
        else:
            guidelines_parts.append(f"{row['Attribute']}: {row['Normalization_instruction']}")

    if attributes_without_instructions:
        guidelines_parts.append(f"{', '.join(attributes_without_instructions)}: Extract as is.")

    return "\n".join(guidelines_parts)


def create_category_guidelines(taxonomy):
    category_guidelines = {}
    for category in taxonomy:
        category_guidelines[category] = get_normalization_guidelines_from_csv(category)
    return category_guidelines


class FewShotExampleSelector:
    def __init__(self, dataset, n_shots=0, selector_type='random'):
        self.dataset = dataset.to_dict('records')
        self.n_shots = n_shots
        self.selector_type = selector_type
        
    def select_examples(self, current_example):
        if self.n_shots == 0:
            return None

        # Filter examples by category using current_example
        category_examples = [
            example for example in self.dataset 
            if example['category'] == current_example['category']
        ]
        
        if self.selector_type == 'semantic':
            # TODO: Implement semantic similarity selection
            pass
        else:
            selected = random.sample(category_examples, min(self.n_shots, len(category_examples)))
        
        # Format examples for prompt creation
        formatted_examples = []
        for example in selected:
            pairs = {}
            for attribute, values in example['target_scores'].items():
                non_na_values = {value for value in values if value != "n/a"}
                pairs[attribute] = list(non_na_values) if non_na_values else "n/a"

            formatted_example = {
                'title': example['input_title'],
                'description': example['input_description'],
                'pairs': pairs
            }
            formatted_examples.append(formatted_example)

        return formatted_examples


def pairs_to_string(pairs):
    formatted_pairs = {
        attribute: values if values else "n/a"
        for attribute, values in pairs.items()
    }
    return json.dumps(formatted_pairs, indent=2)


def create_prompt(
    tokenizer,
    guidelines,
    title,
    description,
    few_shot_examples=None
):
    SYSTEM_PROMPT = "You are a world-class algorithm for extracting information in structured formats."
    INSTRUCTION_TEMPLATE = (
        "Extract the valid attribute values from the product title and description. "
        "Normalize the attribute values according to the guidelines below in JSON format. "
        "Unknown attribute values should be marked as 'n/a'. "
        "Do not explain your answer.\n"
        "Guidelines:\n"
        "{guidelines}"
    )
    INPUT_TEMPLATE = "title: {title}\ndescription: {description}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCTION_TEMPLATE.format(guidelines=guidelines)}
    ]
    
    if few_shot_examples:
        for example in few_shot_examples:
            messages.extend([
                {"role": "user", "content": INPUT_TEMPLATE.format(title=example['title'], description=example['description'])},
                {"role": "assistant", "content": pairs_to_string(example['pairs'])}
            ])
    
    messages.append({
        "role": "user", 
        "content": INPUT_TEMPLATE.format(title=title, description=description)
    })
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
