from .base_trainer import BaseTrainer
from transformers.trainer_utils import EvalLoopOutput
from tqdm import tqdm
import torch


class RetrievalTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _preprocess_output(self, category, logits_list):
        pred_pairs = {k: [] for k in self.taxonomy[category].keys()}
        
        for attribute, attribute_logits in zip(self.taxonomy[category].keys(), logits_list):
            scores = torch.softmax(attribute_logits, dim=0)
            values = [self.model.NULL_VALUE] + list(self.taxonomy[category][attribute])
            pred_value = values[torch.argmax(scores)]
            if pred_value != self.model.NULL_VALUE:
                pred_pairs[attribute].append(pred_value)
                
        return pred_pairs

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="",
    ) -> EvalLoopOutput:
        num_examples = 0
        num_pairs = 0
        self.counter_dict = self._init_counters()
        self.result_lines = []

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        for inputs in tqdm(dataloader):
            outputs = model.infer(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                categories=inputs['categories']
            )
            logits_batch = outputs["logits"]

            for category, true_pairs, logits_list in zip(
                inputs['categories'],
                inputs['pairs_batch'],
                logits_batch
            ):
                if not true_pairs:
                    continue
                    
                num_examples += 1
                pred_pairs = self._preprocess_output(category, logits_list)
                self._evaluate_predictions(category, true_pairs, pred_pairs)
                num_pairs += len(true_pairs)

        metrics = self._compute_metrics(
            num_samples=num_examples,
            num_pairs=num_pairs,
            metric_key_prefix=metric_key_prefix
        )

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_examples
        )
