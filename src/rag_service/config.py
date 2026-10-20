"""Central configuration.

Every knob that changes retrieval quality lives here so a single run can be
described by one object -- that is what gets logged as an MLflow run later.
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    # --- corpus / index locations -------------------------------------------
    source_dir: Path = PROJECT_ROOT / "files"
    corpus_json: Path = PROJECT_ROOT / "data" / "corpus.json"
    index_dir: Path = PROJECT_ROOT / "faiss_ev"
    eval_set: Path = PROJECT_ROOT / "data" / "eval_questions.json"

    # --- experiment tracking -------------------------------------------------
    # A database backend, not ./mlruns: the Model Registry requires one, and
    # MLflow 3 has put the filesystem store into maintenance mode.
    mlflow_tracking_uri: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"

    # --- chunking (tracked per MLflow run) ----------------------------------
    chunk_size: int = 2000
    chunk_overlap: int = 500

    # --- models -------------------------------------------------------------
    # "hashing" (offline), "hf:<model-id>" (local), or an OpenAI model name.
    embedding_model: str = "text-embedding-ada-002"
    hashing_dim: int = 512
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # --- retrieval ----------------------------------------------------------
    retriever_k: int = 4
    score_threshold: float = 0.01

    # secrets come from the unprefixed environment
    openai_api_key: str | None = None

    def model_post_init(self, __context: object) -> None:
        if self.openai_api_key is None:
            import os

            self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def as_run_params(self) -> dict[str, object]:
        """The subset worth logging as MLflow params."""
        return {
            "chunk_size": self.chunk_size,
            "overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "retriever_k": self.retriever_k,
            "score_threshold": self.score_threshold,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
