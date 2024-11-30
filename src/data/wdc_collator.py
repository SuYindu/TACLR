from transformers import DataCollatorWithPadding


class BaseDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        tokenizer_features = [{
            'input_ids': f['input_ids'],
            'attention_mask': f['attention_mask']
        } for f in features]
        batch = super().__call__(tokenizer_features)
        
        batch['categories'] = [f['category'] for f in features]
        batch['pairs_batch'] = [
            {k: v for k, v in f['pairs'].items() if v is not None}
            for f in features
        ]
        
        return batch
