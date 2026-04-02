import os

os.environ["DISABLE_MODEL"] = "true"

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"


def test_health_human() -> None:
    res = client.get("/api/health/human")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data.get("service_status"), str) and data["service_status"]
    assert isinstance(data.get("generation_status"), str) and data["generation_status"]
    assert isinstance(data.get("model_name"), str) and data["model_name"]
    assert "uptime_human" in data


def test_calendar_day() -> None:
    res = client.get("/api/calendar/day", params={"day": "2026-03-30"})
    assert res.status_code == 200
    data = res.json()
    assert data["date_iso"] == "2026-03-30"
    assert isinstance(data.get("topic_of_day"), str) and len(data["topic_of_day"]) > 0
    assert isinstance(data.get("feasts"), list)
    assert isinstance(data.get("saints"), list)
    assert "source" in data


def test_generate() -> None:
    payload = {
        "topic": "Покаяние",
        "occasion": "Великий пост",
        "audience": "приход",
        "bible_text": "Лк. 15:11-32",
    }
    res = client.post("/api/generate", json=payload)
    assert res.status_code == 200
    assert isinstance(res.headers.get("x-request-id"), str)
    assert isinstance(res.headers.get("x-process-time-ms"), str)
    data = res.json()
    assert "sermon" in data
    assert "outline" in data


def test_generate_with_prompt_only() -> None:
    payload = {
        "prompt": "Составь краткую православную проповедь о покаянии и надежде для прихода.",
    }
    res = client.post("/api/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "sermon" in data
    assert len(data["sermon"]) > 0


def test_generate_requires_prompt_or_topic() -> None:
    res = client.post("/api/generate", json={})
    assert res.status_code == 422
    data = res.json()
    assert data["code"] == "VALIDATION_ERROR"
    assert any("Укажите либо «промт для генерации», либо «тему проповеди»" in msg for msg in data["details"])


def test_analyze_validation_error_is_russian() -> None:
    res = client.post("/api/analyze", json={"text": ""})
    assert res.status_code == 422
    data = res.json()
    assert data["code"] == "VALIDATION_ERROR"
    assert data["error"] == "Некорректные данные запроса."
    assert any("Поле «текст для анализа» слишком короткое" in msg for msg in data["details"])
