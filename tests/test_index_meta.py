import json

from rag_service.config import Settings
from rag_service.ingest.pipeline import (
    INDEX_META_FILE,
    read_index_meta,
    resolve_index_settings,
)


def write_meta(tmp_path, **fields):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / INDEX_META_FILE).write_text(json.dumps(fields), encoding="utf-8")
    return index_dir


def test_settings_adopt_the_model_the_index_was_built_with(tmp_path, capsys):
    """Querying an index with a different model is nonsense or a dim error."""
    index_dir = write_meta(tmp_path, embedding_model="hashing", dim=512)
    settings = Settings(index_dir=index_dir, embedding_model="text-embedding-3-small")

    resolved = resolve_index_settings(settings)

    assert resolved.embedding_model == "hashing"
    assert "was built with" in capsys.readouterr().out


def test_matching_model_is_left_alone(tmp_path, capsys):
    index_dir = write_meta(tmp_path, embedding_model="hashing")
    settings = Settings(index_dir=index_dir, embedding_model="hashing")

    resolved = resolve_index_settings(settings)

    assert resolved.embedding_model == "hashing"
    assert capsys.readouterr().out == ""


def test_other_settings_survive_the_adoption(tmp_path):
    index_dir = write_meta(tmp_path, embedding_model="hashing")
    settings = Settings(index_dir=index_dir, embedding_model="openai-x", retriever_k=9)

    resolved = resolve_index_settings(settings)

    assert resolved.retriever_k == 9


def test_missing_metadata_leaves_settings_unchanged(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    settings = Settings(index_dir=index_dir, embedding_model="text-embedding-3-small")

    assert resolve_index_settings(settings).embedding_model == "text-embedding-3-small"
    assert read_index_meta(settings) is None


def test_corrupt_metadata_is_ignored_rather_than_fatal(tmp_path, capsys):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / INDEX_META_FILE).write_text("{not json", encoding="utf-8")
    settings = Settings(index_dir=index_dir, embedding_model="text-embedding-3-small")

    resolved = resolve_index_settings(settings)

    assert resolved.embedding_model == "text-embedding-3-small"
    assert "Ignoring unreadable" in capsys.readouterr().out


def test_metadata_without_a_model_is_harmless(tmp_path):
    index_dir = write_meta(tmp_path, documents=10)
    settings = Settings(index_dir=index_dir, embedding_model="hashing")

    assert resolve_index_settings(settings).embedding_model == "hashing"
