import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer


class BertEncoder(nn.Module):
    def __init__(self, model_name, projection_dim=None):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        
        if not projection_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(self.config.hidden_size, projection_dim)

    def forward(self, sentences, batch_size=32):
        embedding_list = []
        # Split into batches to avoid GPU OOM
        data_loader = DataLoader(sentences, batch_size=batch_size, collate_fn=list, shuffle=False)

        for batch in data_loader:
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
            ).to(next(self.parameters()).device)
            outputs = self.backbone(**inputs, return_dict=True)
            embeddings = outputs.last_hidden_state[:, 0, :].squeeze(dim=1)  # Use CLS token
            embeddings = self.proj(embeddings)
            embeddings = F.normalize(embeddings, dim=1)
            embedding_list.append(embeddings)

        return torch.cat(embedding_list, dim=0)


class PlmForRetrieval(nn.Module):
    PROMPT_TEMPLATE = "A {category} with {attribute} being {value}"
    NULL_VALUE = "n/a"
    LABEL_POS = 1
    LABEL_NULL = 0
    DEFAULT_TEMPERATURE = 0.07
    DEFAULT_NUM_SAMPLES = 128

    def __init__(
        self, 
        model_name, 
        taxonomy, 
        projection_dim=None, 
        temperature=DEFAULT_TEMPERATURE,
        num_samples=DEFAULT_NUM_SAMPLES,
    ):
        super().__init__()
        
        self.encoder = BertEncoder(model_name, projection_dim)
        self.taxonomy = taxonomy
        self.temperature = temperature
        self.num_samples = num_samples

    def _encode_query(self, input_ids, attention_mask):
        outputs = self.encoder.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        embeddings = outputs.last_hidden_state[:, 0, :]  # Use CLS token
        embeddings = self.encoder.proj(embeddings)
        embeddings = F.normalize(embeddings, dim=1)
        return embeddings
        
    def _encode_key(self, prompts, batch_size=32):
        return self.encoder(prompts, batch_size=batch_size)
        
    def forward(
        self,
        input_ids,
        attention_mask,
        categories,
        pairs_batch,
    ):
        """Forward pass for training"""
        device = self.encoder.backbone.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        q_embeddings = self._encode_query(input_ids, attention_mask)
        q_list, k_list, mask_list, labels = [], [], [], []
        
        for category, pairs, q_embedding in zip(categories, pairs_batch, q_embeddings):
            for attribute, pos_values in pairs.items():                        
                value_list = [self.NULL_VALUE]
                
                if pos_values:
                    value_pos = np.random.choice(pos_values)
                    value_list.append(value_pos)
                    labels.append(self.LABEL_POS)
                else:
                    labels.append(self.LABEL_NULL)
                
                negative_candidates = [
                    value for value in self.taxonomy[category][attribute] 
                    if value not in pos_values
                ]
                
                size = min(self.num_samples - len(value_list), len(negative_candidates))
                if size > 0:
                    value_neg = np.random.choice(negative_candidates, size=size, replace=False)
                    value_list.extend(value_neg)
                
                prompts = [
                    self.PROMPT_TEMPLATE.format(
                        category=category, 
                        attribute=attribute, 
                        value=value
                    )
                    for value in value_list
                ]
                k = self._encode_key(prompts)
                
                q_list.append(q_embedding)
                k_list.append(k)
                mask_list.append(torch.zeros(len(prompts), device=q_embedding.device, dtype=torch.bool))
        
        q = torch.stack(q_list)
        k = pad_sequence(k_list, batch_first=True)
        mask = pad_sequence(mask_list, batch_first=True, padding_value=1)
        
        logits = torch.einsum('nh,nkh->nk', q, k)
        logits = logits / self.temperature
        logits[mask] = -10000.0
        
        labels = torch.tensor(labels, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits
        }

    @torch.no_grad()
    def infer(
        self,
        input_ids,
        attention_mask,
        categories,
    ):
        """Inference method"""
        device = self.encoder.backbone.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        q_embeddings = self._encode_query(input_ids, attention_mask)
        logits_list = []
        
        for category, q_embedding in zip(categories, q_embeddings):
            property_logits = []
            
            for attribute, values in self.taxonomy[category].items():
                prompts = [
                    self.PROMPT_TEMPLATE.format(
                        category=category, 
                        attribute=attribute, 
                        value=value
                    )
                    for value in [self.NULL_VALUE] + list(values)
                ]
                
                k_embeddings = self._encode_key(prompts)
                logits = torch.einsum('h,kh->k', q_embedding, k_embeddings)
                property_logits.append(logits)
                
            logits_list.append(property_logits)
                
        return {"logits": logits_list}
