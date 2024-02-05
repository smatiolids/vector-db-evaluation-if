from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, List
from .search_result import EmbeddingSearchResult

from pydantic import BaseModel


class MetricType(str, Enum):
    L2 = "L2"
    COSINE = "COSINE"
    IP = "IP"


class IndexType(str, Enum):
    HNSW = "HNSW"
    DISKANN = "DISKANN"
    IVFFlat = "IVF_FLAT"
    Flat = "FLAT"
    AUTOINDEX = "AUTOINDEX"
    ES_HNSW = "hnsw"


class DBConfig(ABC, BaseModel):
    """DBConfig contains the connection info of vector database

    Args:
        db_label(str): label to distinguish different types of DB of the same database.

            MilvusConfig.db_label = 2c8g
            MilvusConfig.db_label = 16c64g
            ZillizCloudConfig.db_label = 1cu-perf
    """

    db_label: str = ""


class VectorDB(ABC):
    """Each VectorDB will be __init__ once for one case, the object will be copied into multiple processes.

    In each process, the benchmark cases ensure VectorDB.init() calls before any other methods operations

    insert_embeddings, and, search_embedding, will be timed for each call.

    Examples:
        >>> milvus = Milvus()
        >>> with milvus.init():
        >>>     milvus.insert_embeddings()
        >>>     milvus.search_embedding()
    """

    @abstractmethod
    def __init__(
        self,
        database_name: str,
        vector_dimension: int,
        db_config: Optional[DBConfig] = None,
        **kwargs: Any
    ) -> None:
        """
        Initalizes the VectorDB

        Args:
            database_name (str): The name of the vector database.
            vector_dimension (int): The dimensionality of the vectors.
            db_config (Optional[DBConfig]): Configuration for the vector database.
            **kwargs (Any): Additional keyword arguments for vector database initialization.
        
        """

   
    @abstractmethod
    def insert_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadata: Optional[List[dict]] = None,
        **kwargs: Any,
    ) ->  None:
        """Insert the embeddings to the vector database.

        Args:
            ids(List[str]): list of document ids.
            embeddings(List[List[float]]): list of embedding to add to the vector database.
            metadatas(List[dict]): Optional list of metadatas associated with the texts.
            documents(List[str]): list of texts to add to the vectorstore.
            **kwargs(Any): vector database specific parameters.
        """

    @abstractmethod
    def search_embedding(
        self,
        query: List[float],
        k: int = 100,
        filters: dict = {},
        **kwargs: Any,
    ) -> List[EmbeddingSearchResult]:
        """Get k most similar embeddings to query vector.

        Args:
            query(list[float]): query embedding to look up documents similar to.
            k(int): Number of most similar embeddings to return. Defaults to 100.
            filters(dict, optional): filtering expression to filter the data while searching.
            **kwargs(Any): vector database specific parameters.

        Returns:
            list[EmbeddingSearchResult]: list of k most similar EmbeddingSearchResults to the query embedding.
        """
