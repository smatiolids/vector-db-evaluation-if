import logging
import os
import cassio
from cassio.table import MetadataVectorCassandraTable
from typing import Any, List, Optional, Dict

from .vectordb_api import DBConfig, VectorDB
from .search_result import EmbeddingSearchResult

log = logging.getLogger(__name__)


class AstraConfig(DBConfig):
    token: str
    db_id: str


def default_config():
    return AstraConfig(
        token=os.environ.get("ASTRA_TOKEN", ""),
        db_id=os.environ["ASTRA_DB_ID"]
    )


class AstraDBClient(VectorDB):
    """DataStax AstraDB client for VectorDB."""

    def __init__(
        self,
        db_config: DBConfig = None,
        collection_name: str = "vector_store_benchmark_cql",
        vector_dimension: int = 1536,
        drop_old: bool = False,
        keyspace: str = "default_keyspace",
        **kwargs,
    ):
        self.db_config = db_config if db_config is not None else default_config()

        self.collection_name = collection_name
        self.client = cassio.init(
            database_id=self.db_config.db_id,
            token=self.db_config.token,
            keyspace=keyspace
        )

        self.collection = MetadataVectorCassandraTable(
            table=self.collection_name,
            vector_dimension=vector_dimension,
            primary_key_type="TEXT",
        )

        if drop_old:
            try:
                self.collection.clear()
            except:
                drop_old = False
                log.info(
                    f"AstraDB Table: {self.collection_name}")

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
        batch_size = 16
        ttl_seconds = 600000

        if documents is None:
            documents = [None for _ in ids]
        if metadata is None:
            metadata = [{} for _ in ids]
        #
        ttl_seconds = ttl_seconds or self.ttl_seconds

        for i in range(0, len(ids), batch_size):
            batch_texts = documents[i: i + batch_size]
            batch_embedding_vectors = embeddings[i: i + batch_size]
            batch_ids = ids[i: i + batch_size]
            batch_metadata = metadata[i: i + batch_size]

            futures = [
                self.collection.put_async(
                    row_id=text_id,
                    body_blob=text,
                    vector=embedding_vector,
                    metadata=metadata or {},
                    ttl_seconds=ttl_seconds,
                )
                for text, embedding_vector, text_id, metadata in zip(
                    batch_texts, batch_embedding_vectors, batch_ids, batch_metadata
                )
            ]
            for future in futures:
                future.result()
        return ids

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

        print(query)
        print(k)

        results = self.collection.metric_ann_search(
            vector=query,
            n=k,
            metric="cos",
            metadata=filters
        )

        parsed_results = {
            "ids": [],
            "embeddings": [],
            "documents": [],
            "metadatas": [],
            "similarity": []
        }
        
        for r in results:
            parsed_results["ids"].append(r["row_id"])
            parsed_results["embeddings"].append(r["vector"])
            parsed_results["documents"].append(r["body_blob"])
            parsed_results["metadatas"].append(r["metadata"])
            parsed_results["similarity"].append(r["distance"])

        embedding_search_result = EmbeddingSearchResult(**parsed_results)

        return embedding_search_result
