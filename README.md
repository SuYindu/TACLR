# TACLR: A Scalable and Efficient Retrieval-based Method for Industrial Product Attribute Value Identification

## Quick Start

### Multi-label Classification
```bash
accelerate launch \
    --config_file configs/accelerate_config_4gpu.yaml \
    main.py \
    --config configs/roberta_classification_config.yaml
```

### Zero-shot Generation
```bash
# Qwen-based zero-shot inference
accelerate launch \
    --config_file configs/accelerate_config_4gpu.yaml \
    main.py \
    --config configs/qwen_zero_shot_config.yaml

# LLaMA-based zero-shot inference
accelerate launch \
    --config_file configs/accelerate_config_4gpu.yaml \
    main.py \
    --config configs/llama_zero_shot_config.yaml
```

### Few-shot Generation
```bash
# Qwen-based few-shot learning
accelerate launch \
    --config_file configs/accelerate_config_4gpu.yaml \
    main.py \
    --config configs/qwen_few_shot_config.yaml

# LLaMA-based few-shot learning
accelerate launch \
    --config_file configs/accelerate_config_4gpu.yaml \
    main.py \
    --config configs/llama_few_shot_config.yaml
```

### Fine-tuned Generation
```bash
# Qwen fine-tuning
accelerate launch \
    --config_file configs/accelerate_config_4gpu.yaml \
    main.py \
    --config configs/qwen_fine_tune_config.yaml

# LLaMA fine-tuning
accelerate launch \
    --config_file configs/accelerate_config_4gpu.yaml \
    main.py \
    --config configs/llama_fine_tune_config.yaml
```

### Taxonomy-Aware Contrastive Learning Retrieval
```bash
accelerate launch \
    --config_file configs/accelerate_config_8gpu.yaml \
    main.py \
    --config configs/roberta_retrieval_config.yaml
```
