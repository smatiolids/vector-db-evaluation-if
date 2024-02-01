import logging
import os
from astrapy.db import AstraDB, AstraDBCollection
from typing import Any, List, Optional

from pydantic import SecretStr


from .vectordb_api import DBConfig, VectorDB
from .search_result import EmbeddingSearchResult


log = logging.getLogger(__name__)


class AstraConfig(DBConfig):
    token: SecretStr
    api_url: str


def default_config():
    return AstraConfig(
        token=os.environ.get("ASTRA_API_ENDPOINT", ""),
        api_url=os.environ["ASTRA_TOKEN"]
    )


class AstraDBClient(VectorDB):
    """DataStax AstraDB client for VectorDB."""

    def __init__(
        self,
        db_config: DBConfig | None = None,
        collection_name: str = "vector_store_benchmark",
        vector_dimension: int = 1536,
        drop_old: bool = False,
        keyspace: str = "default_keyspace",
        **kwargs,
    ):
        self.db_config = db_config if db_config is not None else default_config()
        self.collection_name = collection_name
        self.client = AstraDB(
            api_endpoint=self.db_config.api_url,
            token=self.db_config.token,
        )

        self.collection = AstraDBCollection(
            self.collection_name,  astra_db=self.client, dimensions=vector_dimension)

        if drop_old:
            try:
                self.client.delete_collection(
                    self.collection_name)  # Reset the database
            except:
                drop_old = False
                log.info(
                    f"AstraDB Collection: {self.collection_name}")

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

        astraDocs = [
            {"_id": doc_id, "document": doc, "$vector": vector, **metadata}
            for doc_id, vector, doc in zip(ids, embeddings, documents)
        ]

        return self.collection.chunked_insert_many(
            documents=astraDocs,
            chunk_size=20,  # Chunk size set to 20 documents
            concurrency=5,  # Concurrently insert 5 chunks at a time
            partial_failures_allowed=True
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
        results = self.collection.vector_find_one(
            query_embeddings=query, limit=k, filter=filters)

        parsed_results = {
            "ids": (r["_id"] for r in results),
            "embeddings": (r["$vector"] for r in results),
            "documents": (r["document"] for r in results),
            "metadatas": (r["metadata"] for r in results)
        }

        embedding_search_result = EmbeddingSearchResult(**parsed_results)

        return embedding_search_result
