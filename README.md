# Православный Интеллектуальный Ассистент (ВКР Prototype)

Прототип интеллектуального ассистента для:
1. Анализа фрагментов Священного Писания.
2. Генерации черновиков православных проповедей.
3. Работы как веб-сервис (FastAPI + web UI).

Проект построен в логике вашей презентации:
- веб-интерфейс;
- сервер обработки запросов;
- модуль NLP/LLM;
- модуль предобработки;
- механизм ответа с опорой на корпус источников.

## Структура проекта

```text
app/                     # FastAPI + UI
  main.py
  services/
  templates/
  static/
train/                   # Подготовка данных и обучение GPT-2/LoRA
  prepare_dataset.py
  train_lora.py
  eval_perplexity.py
data/
  raw/                   # Исходные тексты (bible/commentaries/sermons)
  processed/             # Готовый jsonl-корпус и split'ы
docs/
  TECHNICAL_SPEC.md      # ТЗ
  FUNCTIONAL_REQUIREMENTS.md
  REMOTE_TRAINING.md
  LARGE_DATASET_PIPELINE.md
scripts/
  ingest_real_corpus.py
  setup_server.sh
  run_training.sh
  start_api.sh
  deploy_remote_train.sh
```

## Быстрый локальный запуск

1. Создайте окружение и установите зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Альтернатива одной командой:

```bash
bash scripts/bootstrap_local.sh
```

2. Создайте `.env`:

```bash
cp .env.example .env
```

3. Подготовьте датасет (для примера уже есть демо-файлы):

```bash
python -m train.prepare_dataset --raw-root data/raw
```

Если хотите автоматически собрать «реальный» корпус из вашего локального архива Библии + православных источников RoyalLib:

```bash
cp data/source_import/remote_sources.example.csv data/source_import/remote_sources.csv
# Добавьте в CSV дополнительные источники (чем больше, тем лучше).
python scripts/ingest_real_corpus.py --manifest data/source_import/remote_sources.csv --continue-on-error

# Опционально: положите дополнительные .txt/.zip в data/source_import/manual/** (sermons/commentaries/bible)
# и повторно запустите ingest_real_corpus.py

python -m train.prepare_dataset --raw-root data/raw \
  --deduplicate \
  --balance-source-types \
  --target-ratios "bible=0.20,commentary=0.35,sermon=0.45" \
  --max-total-rows 30000
```

4. Запустите веб-сервис:

```bash
python main.py
```

5. Откройте в браузере: `http://127.0.0.1:8000`

## Дообучение GPT-2 (LoRA)

### Вариант A: локально (если есть GPU)

```bash
accelerate launch train/train_lora.py \
  --train-file data/processed/train.jsonl \
  --valid-file data/processed/valid.jsonl \
  --output-dir outputs/lora-orthodox-gpt2 \
  --base-model ai-forever/rugpt3small_based_on_gpt2 \
  --max-length 512 \
  --per-device-batch-size 2 \
  --grad-accum-steps 8 \
  --num-epochs 3
```

### Вариант B: удалённый сервер (рекомендуется)
Смотрите подробный гайд: `docs/REMOTE_TRAINING.md`.
Бюджетные варианты и точные команды: `docs/BUDGET_GPU_SERVER_2026-03-28.md`.
Пошагово для Google Colab: `docs/COLAB_TRAINING.md`.

Кратко:

```bash
# На сервере
cd ~/ortodox_ai
bash scripts/setup_server.sh ~/ortodox_ai
BASE_MODEL=ai-forever/rugpt3small_based_on_gpt2 bash scripts/run_training.sh
```

## Подключение обученного адаптера в API

```bash
export BASE_MODEL_NAME=ai-forever/rugpt3small_based_on_gpt2
export LORA_ADAPTER_PATH=outputs/lora-orthodox-gpt2
python main.py
```

Для systemd есть шаблон: `scripts/orthodox_ai.service.example`.

## API endpoints

1. `GET /api/health` - статус сервиса и модели.
2. `POST /api/analyze` - анализ фрагмента Писания.
3. `POST /api/generate` - генерация черновика проповеди.

Пример `POST /api/generate`:

```json
{
  "prompt": "Подготовь проповедь о покаянии в пастырском стиле для молодежи на основе Лк. 15:11-32",
  "topic": "Покаяние и надежда",
  "occasion": "Великий пост",
  "audience": "молодежь прихода",
  "bible_text": "Лк. 15:11-32",
  "temperature": 0.75,
  "top_p": 0.92,
  "max_new_tokens": 380
}
```

Если `prompt` заполнен, генерация выполняется в первую очередь по нему.

## Расширение корпуса

- Основной способ: `data/source_import/remote_sources.csv` (CSV manifest источников).
- Локальный импорт: `data/source_import/manual/sermons`, `.../commentaries`, `.../bible`.
- Поддерживаются `.txt` и `.zip`.
- При большом корпусе включайте балансировку в `prepare_dataset.py`, чтобы стиль проповеди не «растворялся» в библейском корпусе.

## Важные замечания

1. Ответы модели являются черновыми и требуют богословской проверки.
2. Для промышленного применения нужна модерация, расширенная оценка качества и юридическая проверка источников.
3. В репозитории даны только демонстрационные тексты. Для ВКР загрузите полный корпус в `data/raw/*`.
