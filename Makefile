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

test: ## unit + smoke tests
	pytest -q

lint: ## ruff
	ruff check .

clean: ## remove runs and prepared data
	rm -rf out data *.egg-info
