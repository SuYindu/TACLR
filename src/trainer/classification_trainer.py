from tqdm import tqdm
import torch
from transformers.trainer_utils import EvalLoopOutput
from .base_trainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def __init__(self, taxonomy, idx_to_label, threshold, *args, **kwargs):
        super().__init__(taxonomy, *args, **kwargs)
        self.idx_to_label = idx_to_label
        self.threshold = threshold

    def _preprocess_output(self, category, true_pairs, logits):
        pred_pairs = {k: [] for k in true_pairs}
        
        confidence_scores = torch.sigmoid(logits)
        pred_indices = torch.where(confidence_scores > self.threshold)[0]
        pred_scores = confidence_scores[pred_indices]
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_label_ids = pred_indices[sorted_indices].tolist()
        
        for idx in pred_label_ids:
            label_category, label_attribute, label_value = self.idx_to_label[idx]
            assert label_category == category, f"Label category {label_category} does not match {category}"
            assert label_attribute in true_pairs, f"Label attribute {label_attribute} not in {true_pairs}"
            pred_pairs[label_attribute].append(label_value)
            
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
            _, logits_batch, _ = self.prediction_step(model, inputs, prediction_loss_only=False)

            for category, true_pairs, logits in zip(inputs['categories'], inputs['pairs_batch'], logits_batch):
                if not true_pairs:
                    continue
                    
                num_examples += 1
                true_pairs = {k: v for k, v in true_pairs.items() if v is not None}
                pred_pairs = self._preprocess_output(category, true_pairs, logits)
                self._evaluate_predictions(category, true_pairs, pred_pairs)
                num_pairs += len(true_pairs)

        metrics = self._compute_metrics(num_samples=num_examples, num_pairs=num_pairs, metric_key_prefix=metric_key_prefix)
        
        return EvalLoopOutput(
            predictions=None, 
            label_ids=None, 
            metrics=metrics, 
            num_samples=num_examples
        )
