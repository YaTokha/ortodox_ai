# Дообучение в Google Colab (пошагово)

Ниже команды именно для **Google Colab Notebook**.
Если вы открыли вкладку `Terminal` в Colab, используйте команды без `!` и `%`.

## 1) Подготовка Colab
1. Откройте [Google Colab](https://colab.research.google.com/).
2. Включите GPU: `Runtime` -> `Change runtime type` -> `T4 GPU`.
3. Создайте новый блок кода и вставьте:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2) Клонирование проекта

### Вариант A (репозиторий публичный)
```python
!cd /content && git clone https://github.com/YaTokha/ortodox_ai.git
```

### Вариант B (репозиторий приватный)
Сначала в GitHub создайте `Personal Access Token` (classic) с правом `repo`, затем:
```python
import os
token = "PASTE_GITHUB_TOKEN_HERE"
!cd /content && git clone https://{token}@github.com/YaTokha/ortodox_ai.git
```

Перейдите в папку проекта и установите зависимости:
```python
%cd /content/ortodox_ai
!python3 -m pip install -U pip
!python3 -m pip install -r requirements.txt
```

## 3) Проверка GPU
```python
!nvidia-smi
```

## 4) Подготовка датасета
Если в проекте уже есть `data/raw/*`, соберите датасет:
```python
!python3 -m train.prepare_dataset \
  --raw-root data/raw \
  --deduplicate \
  --balance-source-types \
  --target-ratios "bible=0.15,commentary=0.25,sermon=0.60" \
  --max-total-rows 75000 \
  --out-corpus data/processed/corpus_xl.jsonl \
  --out-train data/processed/train_xl.jsonl \
  --out-valid data/processed/valid_xl.jsonl \
  --out-test data/processed/test_xl.jsonl
```

## 5) Запуск дообучения (LoRA)
```python
!accelerate launch train/train_lora.py \
  --train-file data/processed/train_xl.jsonl \
  --valid-file data/processed/valid_xl.jsonl \
  --output-dir outputs/lora-orthodox-gpt2-xl \
  --base-model ai-forever/rugpt3small_based_on_gpt2 \
  --max-length 256 \
  --per-device-batch-size 2 \
  --grad-accum-steps 16 \
  --num-epochs 3 \
  --learning-rate 1e-4 \
  --fp16
```

Для T4 обычно это самый безопасный набор параметров.

## 6) Сохранение адаптера на Google Drive
```python
!mkdir -p /content/drive/MyDrive/orthodox_ai_checkpoints
!cp -r outputs/lora-orthodox-gpt2-xl /content/drive/MyDrive/orthodox_ai_checkpoints/
```

## 7) Скачивание адаптера на локальный компьютер
```python
!cd outputs && zip -r lora-orthodox-gpt2-xl.zip lora-orthodox-gpt2-xl
```
После этого скачайте `outputs/lora-orthodox-gpt2-xl.zip` через файловый менеджер Colab.

## 8) Подключение адаптера в локальном проекте
В `.env`:
```env
BASE_MODEL_NAME=ai-forever/rugpt3small_based_on_gpt2
LORA_ADAPTER_PATH=outputs/lora-orthodox-gpt2-xl
```

Запуск:
```bash
python3 main.py
```

## Частые ошибки и решения
1. `-bash: !pip: event not found`
Причина: команда запущена в терминале bash, а не в notebook-ячейке.
Решение: в notebook используйте `!pip ...`, в терминале используйте `pip ...`.

2. `ModuleNotFoundError: No module named 'train'`
Причина: вы не в корне проекта.
Решение: выполните `%cd /content/ortodox_ai` и только потом `python3 -m train.prepare_dataset`.

3. `SyntaxError` на `git clone ...`
Причина: команда написана в Python-ячейке без `!`.
Решение: `!git clone ...`.
