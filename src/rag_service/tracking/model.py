"""The MLflow model wrapping one chunking configuration.

Registering a *model* rather than a row of numbers means the winning config is
reproducible: the artifact carries the FAISS index that config produced, so the
registered version answers queries exactly as the sweep measured it.
"""

import json
from pathlib import Path

import mlflow.pyfunc


class RetrieverModel(mlflow.pyfunc.PythonModel):
    """Takes questions, returns the citations this config retrieves."""

    def load_context(self, context):
        from langchain_community.vectorstores import FAISS

        from rag_service.config import Settings
        from rag_service.ingest.embeddings import get_embeddings

        self.config = json.loads(Path(context.artifacts["config"]).read_text(encoding="utf-8"))
        self.settings = Settings(**self.config)
        self.vectorstore = FAISS.load_local(
            context.artifacts["index"],
            embeddings=get_embeddings(self.settings),
            allow_dangerous_deserialization=True,
        )

    def predict(self, context, model_input, params=None):
        from rag_service.ingest.citations import format_citation

        if hasattr(model_input, "to_dict"):
            questions = [row["question"] for row in model_input.to_dict("records")]
        else:
            questions = list(model_input)

        k = self.settings.retriever_k
        return [
            [
                format_citation(doc.metadata)
                for doc in self.vectorstore.similarity_search(question, k=k)
            ]
            for question in questions
        ]
