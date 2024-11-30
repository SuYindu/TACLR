import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from accelerate import PartialState
from peft import get_peft_model, LoraConfig, TaskType


class LlmForGeneration(nn.Module):
    def __init__(self, tokenizer, model_name, args):
        super().__init__()
        self.tokenizer = tokenizer

        device_string = PartialState().process_index
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={'':device_string},
            torch_dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN"),
        )
        
        # Apply LoRA only in training mode
        if args.do_train:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                bias="none",
                inference_mode=False,
            )
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()
    
    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        **kwargs # for compatibility with other models
    ):
        device = next(self.llm.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Training mode
        if labels is not None:
            labels = labels.to(device)
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False
            )
            return {
                "loss": outputs.loss,
                "logits": outputs.logits
            }

        # Inference mode    
        outputs = self.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
            top_p=None,
            top_k=None,
            temperature=None,
        )
        
        input_length = input_ids.shape[1]
        outputs = outputs[:, input_length:]
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {"generated_text": generated_texts}
