.PHONY: install run test prepare train eval

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run:
	. .venv/bin/activate && python main.py

prepare:
	. .venv/bin/activate && python -m train.prepare_dataset --raw-root data/raw

train:
	. .venv/bin/activate && accelerate launch train/train_lora.py --base-model ai-forever/rugpt3small_based_on_gpt2

eval:
	. .venv/bin/activate && python -m train.eval_perplexity --base-model ai-forever/rugpt3small_based_on_gpt2 --adapter-path outputs/lora-orthodox-gpt2

test:
	. .venv/bin/activate && pytest -q
