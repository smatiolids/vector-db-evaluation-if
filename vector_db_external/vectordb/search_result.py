from pydantic import BaseModel

from typing import Tuple, List, Optional


class EmbeddingSearchResult(BaseModel):
    ids: List[str]
    embeddings: List[float] | None
    metadatas: List[dict]  | List[None] | None
    documents:  List[str]  | List[None | str] | None
