import re
import json
from tqdm import tqdm
from transformers.trainer_utils import EvalLoopOutput
from .base_trainer import BaseTrainer


class GenerationTrainer(BaseTrainer):
    def __init__(self, processor=None, **kwargs):
        super().__init__(**kwargs)
        # processor is a tokenizer
        self.processor = processor

    def _extract_json_text(self, generated_text):
        if "```json" in generated_text:
            json_text = generated_text.split("```json")[1].split("```")[0]
        elif "```" in generated_text:
            json_text = generated_text.split("```")[1].split("```")[0]
        elif "{" in generated_text and "}" in generated_text:
            start = generated_text.find("{")
            end = generated_text.rfind("}") + 1
            json_text = generated_text[start:end]
        else:
            return None

        json_text = re.sub(r':\s*n/a([,}\n])', r': "n/a"\1', json_text)
        return json_text

    def _preprocess_output(self, generated_text):
        json_text = self._extract_json_text(generated_text)
        if json_text is None:
            return None

        try:
            raw_pairs = json.loads(json_text)
        except json.JSONDecodeError:
            return None            
        if not isinstance(raw_pairs, dict):
            return None

        pred_pairs = {}
        for attribute, values in raw_pairs.items():
            if not isinstance(values, list):
                values = [values]
            pred_pairs[attribute] = [str(v) for v in values if v != "n/a"]

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
        self.conversations = []

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        for inputs in tqdm(dataloader):
            prompts = self.processor.batch_decode(inputs['input_ids'])
            outputs = self.model(**inputs)
            generated_texts = outputs["generated_text"]

            for idx, (category, true_pairs, generated_text) in enumerate(
                zip(inputs['categories'], inputs['pairs_batch'], generated_texts)
            ):
                if not true_pairs:
                    continue
                
                num_examples += 1
                self.conversations.extend([
                    f"{'='*80}",
                    f"Sample #{num_examples:03d}",
                    f"{'='*80}",
                    "[Input]",
                    f"{prompts[idx]}",
                    f"\n{'='*80}\n"
                    "[Output]",
                    f"{generated_text}",
                    f"\n{'='*80}\n"
                ])

                pred_pairs = self._preprocess_output(generated_text)
                if pred_pairs is None:
                    self.result_lines.append(f'PARSE ERROR\n\n{generated_text}\n\n')
                    pred_pairs = {attribute: [] for attribute in true_pairs.keys()}
                self._evaluate_predictions(category, true_pairs, pred_pairs)
                num_pairs += len(true_pairs)

        metrics = self._compute_metrics(num_samples=num_examples, num_pairs=num_pairs, metric_key_prefix=metric_key_prefix)

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_examples
        )
