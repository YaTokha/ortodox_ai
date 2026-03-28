# Сбор Большого Корпуса Для Проповедей

## 1) Подготовьте manifest источников

```bash
cp data/source_import/remote_sources.example.csv data/source_import/remote_sources.csv
```

Формат CSV:
- `enabled` (`1/0`)
- `slug`
- `category` (`sermons`, `commentaries`, `bible`)
- `title`
- `author`
- `reference`
- `url`

## 2) Запустите массовый импорт

```bash
python scripts/ingest_real_corpus.py \
  --manifest data/source_import/remote_sources.csv \
  --continue-on-error
```

Опционально:
- `--skip-bible`
- `--only-manifest`
- `--timeout 90`

## 3) Локальные архивы (без ссылок)

Положите файлы в:
- `data/source_import/manual/sermons`
- `data/source_import/manual/commentaries`
- `data/source_import/manual/bible`

Поддерживаются `.txt` и `.zip`.

## 4) Подготовьте обучающий корпус с балансировкой

```bash
python -m train.prepare_dataset --raw-root data/raw \
  --deduplicate \
  --balance-source-types \
  --target-ratios "bible=0.20,commentary=0.35,sermon=0.45" \
  --max-total-rows 30000
```

Рекомендуемый диапазон:
- `20k-60k` примеров (chunks) для `ruGPT3small + LoRA`.

## 5) Обучение

```bash
python train/train_lora.py \
  --train-file data/processed/train.jsonl \
  --valid-file data/processed/valid.jsonl \
  --output-dir outputs/lora-orthodox-gpt2-large \
  --base-model ai-forever/rugpt3small_based_on_gpt2 \
  --max-length 384 \
  --per-device-batch-size 2 \
  --grad-accum-steps 8 \
  --num-epochs 3 \
  --fp16
```
