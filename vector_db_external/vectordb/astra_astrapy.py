import logging
import os
from astrapy.db import AstraDB, AstraDBCollection
from typing import Any, List, Optional, Dict

from .vectordb_api import DBConfig, VectorDB
from .search_result import EmbeddingSearchResult

log = logging.getLogger(__name__)

class AstraConfig(DBConfig):
    token: str
    api_url: str


def default_config():
    return AstraConfig(
        token=os.environ.get("ASTRA_TOKEN", ""),
        api_url=os.environ["ASTRA_API_ENDPOINT"]
    )


class AstraDBClient(VectorDB):
    """DataStax AstraDB client for VectorDB."""

    def __init__(
        self,
        db_config: DBConfig = None,
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

        self.collection = self.client.create_collection(
            collection_name=self.collection_name, dimension=vector_dimension)

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
        ids: List[str],
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
            metadata(list[dict[str, str]]): dict of key and value for metadata
        """

        astraDocs = [
            {"_id": doc_id, "$vector": vector}
            for doc_id, vector in zip(ids, embeddings)
        ]

        if documents is not None:
            list(map(lambda doc, rec: rec.update(
                {"document": doc}), documents, astraDocs))

        if metadata is not None:
            list(map(lambda metadt, rec: rec.update(
                {"metadata": metadt}), metadata, astraDocs))

        return self.collection.chunked_insert_many(
            documents=astraDocs,
            chunk_size=20,  # Chunk size set to 20 documents
            concurrency=5,  # Concurrently insert 5 chunks at a time
            partial_failures_allowed=True
        )

    def search_embedding(
        self,
        query: List[float],
        k: int = 10,
        filters: Dict[str, str] = {},
        **kwargs: Any,
    ) -> List[EmbeddingSearchResult]:
        """Search embeddings from the database.

        Args:
            query(list[float]): embedding to use as a search query
            k(int): number of results to return
            filters(dict[str, str]): dict of key and value for filtering on metadata
            kwargs: other arguments

        """
        metadata_filter = {}

        if bool(filters):
            metadata_filter = {"metadata." + k: v for k, v in filters.items()}

        results = self.collection.vector_find(
            vector=query, limit=k, filter=metadata_filter)

        parsed_results = {
            "ids": (r["_id"] for r in results),
            "embeddings": (r["$vector"] for r in results),
            "documents": (r.get("document", None) for r in results),
            "metadatas": (r.get("metadata", None) for r in results)
        }

        embedding_search_result = EmbeddingSearchResult(**parsed_results)

        return embedding_search_result
