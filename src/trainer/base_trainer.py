from typing import Dict
import math
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from accelerate.data_loader import prepare_data_loader
from datasets import Dataset
from .counter import CustomCounter


MEASUREMENT_ATTRIBUTES = ['Width', 'Height', 'Depth', 'Length']


class BaseTrainer(Trainer):
    def __init__(self, taxonomy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.taxonomy = taxonomy
        self.counter_dict = self._init_counters()
        self.result_lines = []

    def get_train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=True
        )
        train_dataloader = prepare_data_loader(train_dataloader, dispatch_batches=False)

        return train_dataloader

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        # Add placeholder samples to ensure it can be evenly distributed across GPUs
        # This padding is necessary for accurate metric computation in distributed settings
        total_size = len(eval_dataset)
        world_size = self.accelerator.num_processes
        micro_batch_size = self.args.eval_batch_size
        macro_batch_size = micro_batch_size * world_size
        target_size = math.ceil(total_size / macro_batch_size) * macro_batch_size
        num_padding = target_size - total_size
        
        if num_padding > 0:
            placeholder_sample = eval_dataset[-1].copy()
            placeholder_sample['pairs'] = {}
            
            dataset_dicts = eval_dataset.to_list()
            dataset_dicts.extend([placeholder_sample] * num_padding)
            eval_dataset = Dataset.from_list(dataset_dicts)
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        eval_dataloader = prepare_data_loader(eval_dataloader, dispatch_batches=False)
        
        return eval_dataloader

    def _init_counters(self) -> Dict:
        counter_dict = {
            'overall': {
                'average': CustomCounter(),
                'exclude': CustomCounter(),
            },
        }

        for category, attributes in self.taxonomy.items():
            counter_dict[category] = {
                'average': CustomCounter(),
                'exclude': CustomCounter(),
            }
            for attribute in attributes:
                counter_dict['overall'][attribute] = CustomCounter()
                counter_dict[category][attribute] = CustomCounter()

        return counter_dict

    def _compute_metrics(self, num_samples, num_pairs, metric_key_prefix):
        num_samples_tensor = torch.tensor(num_samples, device=self.accelerator.device)
        num_pairs_tensor = torch.tensor(num_pairs, device=self.accelerator.device)
        all_num_samples = self.accelerator.gather_for_metrics(num_samples_tensor).sum().item()
        all_num_pairs = self.accelerator.gather_for_metrics(num_pairs_tensor).sum().item()

        # Gather counter values from all processes
        merged_counter_dict = self._init_counters()
        for category, counters in self.counter_dict.items():
            for attribute, counter in counters.items():
                # Convert to tensors
                tp_tensor = torch.tensor(counter.tp, device=self.accelerator.device)
                fp_tensor = torch.tensor(counter.fp, device=self.accelerator.device)
                fn_tensor = torch.tensor(counter.fn, device=self.accelerator.device)
                support_tensor = torch.tensor(counter.support, device=self.accelerator.device)
                
                # Gather and sum
                all_tp = self.accelerator.gather_for_metrics(tp_tensor).sum().item()
                all_fp = self.accelerator.gather_for_metrics(fp_tensor).sum().item()
                all_fn = self.accelerator.gather_for_metrics(fn_tensor).sum().item()
                all_support = self.accelerator.gather_for_metrics(support_tensor).sum().item()
                
                merged_counter_dict[category][attribute].tp = all_tp
                merged_counter_dict[category][attribute].fp = all_fp
                merged_counter_dict[category][attribute].fn = all_fn
                merged_counter_dict[category][attribute].support = all_support

        # Compute metrics
        metrics = {
            f'{metric_key_prefix}_num_samples': all_num_samples,
            f'{metric_key_prefix}_num_pairs': all_num_pairs,
        }
        
        for counters in merged_counter_dict.values():
            for counter in counters.values():
                counter.compute_scores()

        sorted_categories = sorted(
            merged_counter_dict.keys(),
            key=lambda x: merged_counter_dict[x]['average'].support,
            reverse=True
        )
        
        for category in sorted_categories:
            counters = merged_counter_dict[category]
            sorted_attributes = sorted(
                [attribute for attribute in counters],
                key=lambda x: counters[x].support,
                reverse=True
            )
            
            for attribute in sorted_attributes:
                counter = counters[attribute]
                metrics.update({
                    f'{metric_key_prefix}_{category}_{attribute}_precision': counter.precision * 100,
                    f'{metric_key_prefix}_{category}_{attribute}_recall': counter.recall * 100,
                    f'{metric_key_prefix}_{category}_{attribute}_f1_score': counter.f1_score * 100,
                    f'{metric_key_prefix}_{category}_{attribute}_support': counter.support,
                })

        return metrics

    @staticmethod
    def _format_result(result_type, attribute, true_values, pred_values):
        return f"{result_type:<20}{attribute:<30}{str(true_values):<80}{str(pred_values)}"

    def _evaluate_predictions(self, category, true_pairs, pred_pairs):
        for attribute in true_pairs:
            true_values = true_pairs[attribute]
            pred_values = pred_pairs.get(attribute, [])
            
            tp, fp, fn, tn = 0, 0, 0, 0
            if not true_values and not pred_values:
                result_type = "TRUE_NEGATIVE"
                tn = 1
            elif not true_values and pred_values:
                result_type = "FALSE_POSITIVE"
                fp = 1
            elif true_values and not pred_values:
                result_type = "FALSE_NEGATIVE"
                fn = 1
            else:
                top1_value = pred_values[0]
                if top1_value in true_values:
                    result_type = "TRUE_POSITIVE"
                    tp = 1
                else:
                    result_type = "FALSE_POS_NEG"
                    fp, fn = 1, 1
            
            result_line = self._format_result(result_type, attribute, true_values, pred_values)
            self.result_lines.append(result_line)

            self.counter_dict['overall']['average'].add(tp, fp, fn)
            self.counter_dict['overall'][attribute].add(tp, fp, fn)
            self.counter_dict[category]['average'].add(tp, fp, fn)
            self.counter_dict[category][attribute].add(tp, fp, fn)
            if attribute not in MEASUREMENT_ATTRIBUTES:
                self.counter_dict['overall']['exclude'].add(tp, fp, fn)
                self.counter_dict[category]['exclude'].add(tp, fp, fn)
