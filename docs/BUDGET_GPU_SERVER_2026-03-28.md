# Бюджетный GPU-сервер: выбор и точные команды (снимок на 2026-03-28)

## 1) Актуальные цены (официальные источники)

1. RunPod RTX 4090: от **$0.59/ч** (Secure Cloud) и **$0.34/ч** (Community) — официальная страница модели.
2. Vast.ai RTX 4090 (on-demand marketplace): по публичному API на момент снимка:
   - min: **$0.1356/ч**
   - p50: **$0.3519/ч**
   - verified + reliability >= 0.98: min **$0.2015/ч**, p50 **$0.3919/ч**.
3. Selectel (облачные серверы):
   - GPU A100 40 ГБ: **187.6188 ₽/ч** (136 961.72 ₽/мес)
   - GPU A100 прерываемая: **80.4070 ₽/ч** (58 697.12 ₽/мес)

Источники:
1. https://www.runpod.io/gpu-models/rtx-4090
2. https://docs.vast.ai/api-reference/search/search-offers
3. https://console.vast.ai/api/v0/bundles/
4. https://selectel.ru/prices/

## 2) Рекомендация под диплом (стоимость/качество)

Для вашей задачи (дообучение GPT-2/LoRA + веб-сервис):

1. Экономичный вариант: **Vast.ai RTX 4090 (verified, reliability >= 0.98)**.
2. Стабильный вариант: **RunPod Secure RTX 4090 ($0.59/ч)**.

Практически: если бюджет ограничен и обучение можно перезапускать, берите Vast.ai.
Если нужен более предсказуемый прод-ран и меньше риска прерываний, берите RunPod Secure.

## 3) Точные команды после аренды сервера

Предположим, вы уже получили `SERVER_IP` и SSH-доступ (обычно `root@SERVER_IP`).

### 3.1 Деплой проекта и запуск обучения

```bash
cd /Users/tuhtaevtahir/Desktop/project/ortodox_ai
bash scripts/deploy_remote_train.sh root@SERVER_IP /root/ortodox_ai
```

Что делает скрипт:
1. Синхронизирует проект на сервер.
2. Устанавливает зависимости (`scripts/setup_server.sh`).
3. Запускает обучение (`scripts/run_training.sh`).

### 3.2 Ручной запуск API на сервере

```bash
ssh root@SERVER_IP
cd /root/ortodox_ai
BASE_MODEL_NAME=ai-forever/rugpt3small_based_on_gpt2 \
LORA_ADAPTER_PATH=outputs/lora-orthodox-gpt2 \
bash scripts/start_api.sh
```

Проверка:

```bash
curl http://SERVER_IP:8000/api/health
```

## 4) Минимальные настройки инстанса

Для RunPod/Vast.ai ставьте:
1. GPU: `1x RTX 4090 (24 GB)`.
2. CPU: `>= 8 vCPU`.
3. RAM: `>= 32 GB`.
4. Disk: `>= 120 GB` (лучше 200 GB).
5. OS: `Ubuntu 22.04`.
6. Открытые порты: `22` и `8000`.

## 5) Быстрый расчёт бюджета

1. Стоимость обучения = `ставка_в_час * часы`.
2. Пример 12 часов:
   - Vast verified min: `0.2015 * 12 ≈ $2.42`
   - Vast verified p50: `0.3919 * 12 ≈ $4.70`
   - RunPod Secure: `0.59 * 12 = $7.08`
