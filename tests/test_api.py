"""FastAPI uç nokta testleri."""
from fastapi.testclient import TestClient

import api
import database


def test_health_and_search(tmp_path, monkeypatch):
    db_file = tmp_path / "api.db"
    conn = database.get_connection(str(db_file))
    database.seed_if_empty(conn)
    monkeypatch.setattr(api, "_conn", conn)

    client = TestClient(api.app)
    assert client.get("/api/health").json()["status"] == "ok"

    recipes = client.get("/api/recipes")
    assert recipes.status_code == 200
    assert len(recipes.json()) >= 1

    search = client.get("/api/search", params={"q": "menemen"})
    assert search.status_code == 200
    body = search.json()
    assert body["recipes"]
    assert "Menemen" in body["recipes"][0]["title"]
    assert "yumurta" in body["answer"].lower() or "Menemen" in body["answer"]

    vegan = client.get("/api/search", params={"q": "Menemen vegan mı?"})
    assert vegan.status_code == 200
    assert "vegan değil" in vegan.json()["answer"].lower() or "vegan degil" in vegan.json()["answer"].lower()

    empty = client.get("/api/search", params={"q": ""})
    assert empty.status_code == 200
    assert "Boş soru" in empty.json()["answer"]

    match = client.post("/api/match", json={"ingredients": "yumurta, un, sut"})
    assert match.status_code == 200
    assert isinstance(match.json(), list)
