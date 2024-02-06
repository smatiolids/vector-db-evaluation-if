from pydantic import BaseModel

from typing import List, Union
NoneType = type(None)

class EmbeddingSearchResult(BaseModel):
    ids: List[str]
    embeddings: List[List[float]]
    metadatas: Union[List[Union[dict, NoneType]], None]
    documents:  Union[List[Union[str, NoneType]], None]
