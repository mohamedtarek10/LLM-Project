"""Request/response contracts for the HTTP API."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

# An empty or whitespace-only question is a client error, not an empty answer:
# strip first, then require at least one character -> FastAPI returns 422.
NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
]


class AskRequest(BaseModel):
    question: NonEmptyText
    session_id: str = Field(
        default="default",
        description="Opaque conversation id; omit for a stateless question.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"question": "What does the contract say about delivery?"}]
        }
    }


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(
        default_factory=list,
        description="Citations for the passages used, e.g. 'Article 418'. Never chunk ids.",
    )


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
