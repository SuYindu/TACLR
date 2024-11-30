import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class PlmForClassification(nn.Module):
    def __init__(self, model_name, idx_to_label, label_to_idx):
        super().__init__()
        self.idx_to_label = idx_to_label
        self.label_to_idx = label_to_idx
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(idx_to_label),
            problem_type="multi_label_classification"
        )
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        input_ids,
        attention_mask,
        categories,
        pairs_batch,
        labels_batch=None,
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        base_logits = outputs.logits

        # mask for the valid pairs
        all_logits = []
        batch_loss = None if labels_batch is None else 0.0
        for i, (category, pairs) in enumerate(zip(categories, pairs_batch)):
            sample_mask = torch.zeros(len(self.label_to_idx), device=base_logits.device)
            valid_pairs = {(category, attribute) for attribute in pairs}
            
            for (tuple_category, tuple_attribute, _), idx in self.label_to_idx.items():
                if (tuple_category, tuple_attribute) in valid_pairs:
                    sample_mask[idx] = 1
            
            sample_logits = base_logits[i] * sample_mask - (1 - sample_mask) * 1e5
            all_logits.append(sample_logits)
            
            if labels_batch is not None:
                sample_loss = self.bce_loss(sample_logits, labels_batch[i])
                sample_loss = (sample_loss * sample_mask).sum() / (sample_mask.sum() + 1e-6)
                batch_loss += sample_loss
        
        stacked_logits = torch.stack(all_logits)
        if labels_batch is not None:
            batch_loss = batch_loss / len(input_ids)

        return {
            "logits": stacked_logits,
            "loss": batch_loss
        }
