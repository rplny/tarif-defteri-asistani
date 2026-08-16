"""Testler için geçici SQLite bağlantısı."""
import os

os.environ["FOUNDRY_RAG"] = "0"

import database
import pytest


@pytest.fixture()
def conn(tmp_path):
    db_file = tmp_path / "test_recipes.db"
    connection = database.get_connection(str(db_file))
    database.seed_if_empty(connection)
    yield connection
    connection.close()
