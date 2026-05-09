# finetune

A minimal LoRA fine-tune of `Qwen/Qwen2.5-3B-Instruct` on a tiny custom Q&A dataset, run end-to-end in a Jupyter notebook. The fine-tuned adapter + merged tokenizer is saved to `./my-qwen` and reloaded with a `transformers` pipeline for inference.

## Layout

```
LLM Fine Tuning.ipynb  # full pipeline: load -> tokenize -> LoRA -> train -> save -> infer
christian.json         # 3-row prompt/completion training set
my-qwen/               # output dir for the fine-tuned model (created by the notebook)
trainer_output/        # HF Trainer checkpoints (gitignored)
```

## What the notebook does

1. Loads `christian.json` as a HuggingFace `Dataset` (3 prompt/completion rows).
2. Tokenizes with the Qwen tokenizer, padding to length 128, copying `input_ids` to `labels` for causal LM.
3. Wraps `Qwen2.5-3B-Instruct` (loaded in fp16 on CUDA) with a PEFT `LoraConfig` targeting `q_proj`, `k_proj`, `v_proj`.
4. Trains with HF `Trainer` for 100 epochs, `lr=1e-3`, fp16.
5. Saves the adapter + tokenizer to `./my-qwen`.
6. Reloads it as a `transformers.pipeline(...)` and asks a few questions about the training subject.

## Requirements

A CUDA GPU and:

```bash
pip install transformers datasets peft accelerate torch
```

## Run

```bash
jupyter lab "LLM Fine Tuning.ipynb"
```

Then run cells top-to-bottom. The `./my-qwen` directory will hold the trained model after the save cell.

## Notes

- The dataset is tiny by design — it's a teaching example for LoRA mechanics, not a serious fine-tune. With 3 rows and 100 epochs you will memorise rather than generalise.
- To swap in your own data, replace `christian.json` with the same `{"prompt": ..., "completion": ...}` schema.
