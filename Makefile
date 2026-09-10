.PHONY: help install data train eval generate test lint clean

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## install the package in editable mode (CPU torch)
	pip install -e ".[dev]"

data: ## prepare the synthetic smoke corpus
	python scripts/prepare_data.py --source synthetic --target-chars 200000 --out data/synthetic

train: ## CPU smoke run
	python scripts/train.py --config configs/cpu_smoke.json

train-gpu: ## single-GPU run (edit configs/gpu_1x.json for your data first)
	python scripts/train.py --config configs/gpu_1x.json

eval: ## evaluate the best CPU checkpoint
	python scripts/evaluate.py --ckpt out/cpu-smoke/best --data data/synthetic.bin --sample 200

generate: ## sample from the best CPU checkpoint
	python scripts/generate.py --ckpt out/cpu-smoke/best --tokenizer data/synthetic.tokenizer.json --prompt "the quiet cat"

tokdata: ## write the tokenizer research corpus
	python scripts/tokenizer_prepare_corpus.py --out data/tokenizer/indic-v1

toktrain: ## train the HuggingFace BPE baseline (needs pip install ".[tokenizer]")
	python scripts/tokenizer_train.py --corpus data/tokenizer/indic-v1 --impl bpe_hf \
		--vocab-size 1024 --out artifacts/tokenizers/bpe_hf_1024 --exp-id EXP-002

tokcompare: ## compare every trained tokenizer artifact on the probe corpus
	python scripts/tokenizer_compare.py --corpus data/tokenizer/indic-v1 \
		--tokenizer artifacts/tokenizers/char artifacts/tokenizers/word \
		            artifacts/tokenizers/bpe_py_512 artifacts/tokenizers/bpe_py_1024 \
		            artifacts/tokenizers/bpe_hf_1024 \
		--out out/tokenizer/compare.json --exp-id EXP-002

test: ## unit + smoke tests
	pytest -q

lint: ## ruff
	ruff check .

clean: ## remove runs, prepared data and tokenizer artifacts
	rm -rf out data artifacts *.egg-info
