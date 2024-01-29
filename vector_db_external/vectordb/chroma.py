import logging
import os
import chromadb
from typing import Any, List, Optional

from pydantic import SecretStr


from .vectordb_api import DBConfig, VectorDB
from .search_result import EmbeddingSearchResult


log = logging.getLogger(__name__)


class ChromaConfig(DBConfig):
    password: SecretStr
    host: str
    port: str


def default_config():
    return ChromaConfig(
        password=os.environ.get("CHROMA_SERVER_PASSWORD", ""),
        host=os.environ["CHROMA_SERVER_HOST"],
        port=os.environ["CHROMA_SERVER_HTTP_PORT"],
    )


class ChromaClient(VectorDB):
    """Chroma client for VectorDB."""

    def __init__(
        self,
        database_name: str = "vector_store_benchmark",
        vector_dimension: int = 1536,
        db_config: DBConfig | None = None,
        drop_old: bool = False,
        client_mode: str = "server",
        database_path: str = None,
        **kwargs,
    ):
        self.db_config = db_config if db_config is not None else default_config()
        self.collection_name = database_name
        if client_mode == "server":
            self.client = chromadb.HttpClient(
                host=self.db_config.host, port=self.db_config.port
            )
        else:
            self.client = chromadb.PersistentClient(path=database_path)


        self.collection = self.client.get_or_create_collection(self.collection_name)

        if self.client.heartbeat() is None:
            raise ConnectionError

        if drop_old:
            try:
                self.client.reset()  # Reset the database
            except:
                drop_old = False
                log.info(f"Chroma client drop_old collection: {self.collection_name}")

    def insert_embeddings(
        self,
        ids: list[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadata: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> None:
        """Insert embeddings into the database.

        Args:
            embeddings(list[list[float]]): list of documents' embeddings
            documents(list[str]): list of textual documents
            ids(list[str]): list of ids for each given document
            metadata(dict[str, str]): dict of key and value for metadata
        """
        return self.collection.add(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata,
            documents=documents,
            **kwargs,
        )

    def search_embedding(
        self,
        query: list[float],
        k: int = 10,
        filters: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> List[EmbeddingSearchResult]:
        """Search embeddings from the database.

        Args:
            query(list[float]): embedding to use as a search query
            k(int): number of results to return
            filters(dict[str, str]): dict of key and value for filtering on metadata
            kwargs: other arguments

        """
        results = self.collection.query(query_embeddings=query, n_results=k, where=filters)

        if results["embeddings"] is not None:
            embeddings =  results["embeddings"][0]
        else:
            embeddings = None

        if results["documents"] is not None:
            documents =  results["documents"][0]
        else:
            documents = None

        if results["metadatas"] is not None:
            metadatas =  results["metadatas"][0]
        else:
            metadatas = None


        parsed_result = {
            "ids": results["ids"][0],
            "embeddings": embeddings,
            "documents": documents,
            "metadatas": metadatas
        }

        embedding_search_result = EmbeddingSearchResult(**parsed_result)

        return embedding_search_result
