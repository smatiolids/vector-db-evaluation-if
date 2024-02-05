from pydantic import BaseModel

from typing import Tuple, List, Optional, Union


class EmbeddingSearchResult(BaseModel):
    ids: List[str]
    embeddings: List[List[float]]
    metadatas: Union[List[dict], None]
    documents:  Union[List[str], None]
