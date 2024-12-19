import torch
from .wdc_collator import BaseDataCollator


def create_label_mapping(taxonomy):
    unique_tuples = {
        (category, attribute, value)
        for category, attributes in taxonomy.items()
        for attribute, values in attributes.items()
        for value in values
    }
    unique_tuples = sorted(unique_tuples)
    idx_to_label = {idx: label_tuple for idx, label_tuple in enumerate(unique_tuples)}
    label_to_idx = {label_tuple: idx for idx, label_tuple in enumerate(unique_tuples)}
    return idx_to_label, label_to_idx


def create_label_tensor(category, pairs, label_to_idx):
    labels = torch.zeros(len(label_to_idx), dtype=torch.float)

    for attribute, values in pairs.items():
        for value in values:
            label_tuple = (category, attribute, value)
            assert label_tuple in label_to_idx, f"Label tuple {label_tuple} not found in label_to_idx"
            label_idx = label_to_idx[label_tuple]
            labels[label_idx] = 1.0

    return labels


class ClassificationDataCollator(BaseDataCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        if 'labels' in features[0]:
            batch['labels_batch'] = torch.stack([torch.tensor(f['labels']) for f in features])
        return batch
