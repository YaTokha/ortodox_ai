# Обучение на удалённом сервере (GPU)

## 1. Требования к серверу
1. Ubuntu 22.04+
2. NVIDIA GPU (желательно 16-24 GB VRAM и выше)
3. Python 3.10+
4. Доступ по SSH

## 2. Копирование проекта
Локально выполните:

```bash
rsync -avz --exclude '.venv' --exclude '__pycache__' /path/to/ortodox_ai/ user@SERVER_IP:~/ortodox_ai/
```

## 3. Подготовка окружения на сервере
```bash
ssh user@SERVER_IP
cd ~/ortodox_ai
bash scripts/setup_server.sh ~/ortodox_ai
```

## 4. Подготовка корпуса
1. Скопируйте ваши реальные тексты в `data/raw/bible`, `data/raw/commentaries`, `data/raw/sermons`.
2. Проверьте, что файлы в UTF-8.

## 5. Запуск обучения
```bash
cd ~/ortodox_ai
BASE_MODEL=ai-forever/rugpt3small_based_on_gpt2 \
OUTPUT_DIR=outputs/lora-orthodox-gpt2 \
NUM_EPOCHS=3 \
MAX_LENGTH=256 \
PER_DEVICE_BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=16 \
LEARNING_RATE=1e-4 \
bash scripts/run_training.sh
```

Или одной командой с локальной машины:

```bash
cd /Users/tuhtaevtahir/Desktop/project/ortodox_ai
NUM_EPOCHS=3 MAX_LENGTH=256 PER_DEVICE_BATCH_SIZE=1 GRAD_ACCUM_STEPS=16 LEARNING_RATE=1e-4 \
bash scripts/deploy_remote_train.sh root@SERVER_IP /root/ortodox_ai
```

## 6. Проверка результата
После обучения проверьте:
1. `outputs/lora-orthodox-gpt2/adapter_model.safetensors`
2. `outputs/lora-orthodox-gpt2/metrics.json`

## 7. Запуск сервиса с адаптером
```bash
cd ~/ortodox_ai
BASE_MODEL_NAME=ai-forever/rugpt3small_based_on_gpt2 \
LORA_ADAPTER_PATH=outputs/lora-orthodox-gpt2 \
bash scripts/start_api.sh
```

Сервис поднимется на `http://SERVER_IP:8000`.

## 8. Опционально: запуск в фоне (systemd)
Создайте unit-файл, который стартует `scripts/start_api.sh` от имени пользователя.
