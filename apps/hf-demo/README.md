---
title: Frontier AI research preview (EXP-043)
emoji: 📝
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# Frontier AI research preview: EXP-043 (190 M parameters)

A base language model trained **from scratch** on 12 Indian languages and English, one pass over
2.78 billion tokens on free Kaggle T4 GPUs. It **continues** a text you type. It cannot answer
questions, follow instructions or chat yet (Belebele reading test: chance level in 12 of 13
languages). It often writes fluent but wrong text; do not rely on it.

- Scores (bits per byte, lower is better): 0.6283 mean over 13 languages on the validation split of
  its training corpus; 0.9099 on our separate protected held-out suite (EXP-043, EXP-045).
- Architecture: 14 layers, width 896, RoPE, grouped-query attention (2 key/value heads), SwiGLU,
  RMSNorm, context 512 tokens; Frontier Tokenizer v2 (32,896 tokens).
- Weights here: the float32 training result converted to fp16 (`model_fp16.pt`); `MODEL_INFO.json`
  records both SHA-256 values.
- Records of every experiment and decision: https://github.com/dfgtghu556-ops/frontier-ai
  (EXP-043, EXP-045, D-049).

This page is a **private preview**. The model's licence and the data attributions for a public
release have not been decided yet.
