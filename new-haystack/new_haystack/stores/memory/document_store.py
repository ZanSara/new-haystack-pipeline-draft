from typing import Literal, Any, Dict, List, Optional, Union, Iterable

import logging

import numpy as np

from new_haystack.data import Document, Query, TextDocument
from new_haystack.models.device_management import initialize_device_settings

from new_haystack.stores._utils import MissingEmbeddingError
from new_haystack.stores.memory.store import MemoryStore
from new_haystack.stores.memory._bm25 import BM25Representation, BM25RepresentationMissing
from new_haystack.stores.memory._scores import (
    get_scores_numpy,
    get_scores_torch,
    scale_to_unit_interval,
)


logger = logging.getLogger(__name__)

try:
    import torch
except ImportError as e:
    logger.debug(
        "torch not found: MemoryDocumentStore won't be able to search by embedding with a local model."
    )


class MemoryDocumentStore:
    def __init__(
        self,
        use_bm25: bool = True,
        bm25_tokenization_regex: str = r"(?u)\b\w\w+\b",
        bm25_algorithm: Literal["BM25Okapi", "BM25L", "BM25Plus"] = "BM25Okapi",
        bm25_parameters: dict = {},
        use_gpu: bool = True,
        devices: Optional[List[Union[str, "torch.device"]]] = None,
    ):
        self.document_store = MemoryStore(pool="documents")

        # For BM25 retrieval
        self.use_bm25 = use_bm25
        self.bm25_algorithm = (bm25_algorithm,)
        self.bm25_parameters = (bm25_parameters,)
        self.bm25_tokenization_regex = bm25_tokenization_regex
        self.bm25 = {}
        if use_bm25:
            self.bm25 = {
                "documents": BM25Representation(
                    bm25_algorithm=bm25_algorithm,
                    bm25_parameters=bm25_parameters,
                    bm25_tokenization_regex=bm25_tokenization_regex,
                )
            }

        # For embedding retrieval
        self.device = None
        init_devices, _ = initialize_device_settings(
            devices=devices, use_cuda=use_gpu, multi_gpu=False
        )
        if init_devices:
            if len(init_devices) > 1:
                logger.warning(
                    "Multiple devices are not supported in %s inference, using the first device %s.",
                    self.__class__.__name__,
                    init_devices[0],
                )
            self.device = init_devices[0]

    def create_pool(self, pool: str, use_bm25: Optional[bool] = None):
        """
        Creates a new pool with the given name.
        The BM25 representation is initialized according to use_bm25 if given,
        otherwise defaults to the value given at init time.

        :param pool: the pool name
        :param use_bm25: whether to initialize the BM25 representation for this pool
        """
        self.document_store.create_pool(pool=pool)

        if use_bm25 or (use_bm25 is None and self.use_bm25):
            self.bm25[pool] = BM25Representation(
                bm25_algorithm=self.bm25_algorithm,
                bm25_parameters=self.bm25_parameters,
                bm25_tokenization_regex=self.bm25_tokenization_regex,
            )

    def list_pools(self) -> List[str]:
        """
        Returns a list of all the pools present in this store.
        """
        return self.list_pools()

    def delete_pool(
        self, pool: str = "documents", delete_populated_pool: bool = False
    ) -> None:
        """
        Drops a pool completely. Will not delete pools that contains items unless
        `delete_populated_pool=True` (default is False).

        :param pool: the pool to drop
        :param delete_populated_pool: whether to drop full pools too
        :raises PoolFullError if the pool is full and delete_populated_pool=False
        """
        self.document_store.delete_pool(
            pool=pool, delete_populated_pool=delete_populated_pool
        )
        if self.bm25:
            del self.bm25[pool]

    def has_document(self, id: str, pool: str = "documents") -> bool:
        """
        Checks if this ID exists in the document store.

        :param id: the id to find in the document store.
        :param pool: in which pool to look for this document.
        """
        return self.document_store.has_item(id=id, pool=pool)

    def get_document(self, id: str, pool: str = "documents") -> Optional[Document]:
        """
        Finds a document by ID in the document store.

        :param id: the id of the document to get.
        :param pool: in which pool to look for this document.
        """
        return Document.from_dict(self.document_store.get_item(id=id, pool=pool))

    def count_documents(self, filters: Dict[str, Any], pool: str = "documents"):
        """
        Returns the number of how many documents match the given filters.
        Pass filters={} to count all documents in the given pool.

        :param filters: the filters to apply to the documents list.
        :param pool: in which pool to look for this document.
        """
        return self.document_store.count_items(filters=filters, pool=pool)

    def get_document_ids(
        self, filters: Dict[str, Any], pool: str = "documents"
    ) -> Iterable[str]:
        """
        Returns only the ID of the documents that match the filters provided.

        :param filters: the filters to apply to the documents list.
        :param pool: in which pool to look for this document.
        """
        return self.document_store.get_ids(filters=filters, pool=pool)

    def get_documents(
        self, filters: Dict[str, Any], pool: str = "documents"
    ) -> Iterable[Document]:
        """
        Returns the documents that match the filters provided.

        :param filters: the filters to apply to the documents list.
        :param pool: in which pool to look for this document.
        """
        for doc in self.document_store.get_items(filters=filters, pool=pool):
            yield Document.from_dict(dictionary=doc)

    def write_documents(
        self,
        documents: Iterable[Document],
        pool: str = "documents",
        duplicates: Literal["skip", "overwrite", "fail"] = "overwrite",
    ) -> None:
        """
        Writes documents into the store.

        :param documents: a list of Haystack Document objects.
        :param pool: write documents to a different namespace. The default namespace is 'documents'.
        :param duplicates: Documents with the same ID count as duplicates. When duplicates are met,
            Haystack can choose to:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised
        :raises DuplicateDocumentError: Exception trigger on duplicate document
        :return: None
        """
        self.document_store.write_items(
            items=(doc.to_dict() for doc in documents),
            pool=pool,
            duplicates=duplicates,
        )
        if self.bm25 and len(documents) > 0:
            self.bm25[pool].update_bm25(self.get_documents(filters={}, pool=pool))

    def delete_documents(
        self,
        ids: List[str],
        pool: str = "documents",
        fail_on_missing_item: bool = False,
    ) -> None:
        """
        Deletes all ids from the given pool.

        :param ids: the ids to delete
        :param pool: the pool where these id should be stored
        :param fail_on_missing_item: fail if the id is not found, log ignore otherwise
        """
        self.document_store.delete_items(
            ids=ids, pool=pool, fail_on_missing_item=fail_on_missing_item
        )

    def get_relevant_documents(
        self,
        queries: List[Query],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        use_bm25: bool = True,
        similarity: str = "dot_product",
        scoring_batch_size: int = 500000,
        scale_score: bool = True,
        pool: str = "documents",
    ) -> Dict[str, List[Document]]:
        """
        Performs document retrieval, either by BM25, or by embedding, according to the input parameters.

        :param queries: either strings for bm25 retrieval or embeddings for embedding retrieval.
        :param filters: return only documents that match these filters too
        :param top_k: how many documents to return at most. Might return less documents if the filters
            don't return enough documents.
        :param use_bm25: whether to do the retrieval with bm25 or embeddings
        :param pool: the pool to get the documents from.
        :returns: an iterable of Documents that match the filters, ranked by similarity score.
        """
        filters = filters or {}

        # BM25 Retrieval
        if use_bm25:
            relevant_documents = {}
            for query in queries:
                relevant_documents[query.id] = self._bm25_retrieval(
                    query=query,
                    filters=filters,
                    top_k=top_k,
                    pool=pool,
                )
            return relevant_documents

        # Embedding Retrieval
        relevant_documents = self._embedding_retrieval(
            queries=queries,
            filters=filters,
            top_k=top_k,
            pool=pool,
            similarity=similarity,
            batch_size=scoring_batch_size,
            scale_score=scale_score,
        )
        return relevant_documents

    def _bm25_retrieval(
        self,
        query: Query,
        filters: Dict[str, Any],
        top_k: int,
        pool: str,
    ) -> List[Document]:
        """
        Performs BM25 retrieval using the rank_bm25 pools.
        """
        if query is None or not query.content:
            logger.info(
                "You tried to perform retrieval on an empty query. No documents returned for it."
            )
            return []

        if pool not in self.bm25.keys():
            raise BM25RepresentationMissing(
                f"No BM25 representation for pool {pool}"
            )  # TODO add a way to create such pool

        filtered_document_ids = (
            self.get_document_ids(
                filters={**filters, "content_type": "text"}, pool=pool
            ),
        )
        tokenized_query = self.bm25[pool].bm25_tokenization_regex(query.content.lower())
        docs_scores = self.bm25[pool].bm25.get_scores(tokenized_query)
        most_relevant_ids = np.argsort(docs_scores)[::-1]

        # We're iterating this way to avoid consuming the incoming iterator
        # We'd should keep everything as lazy as possible (no len(), no
        # direct item access, ...).
        current_position = 0
        returned_docs = 0
        while returned_docs < top_k:
            try:
                id = most_relevant_ids[current_position]
            except IndexError as e:
                logging.debug(
                    f"Returning less than top_k results as the filters returned less than {top_k} documents."
                )
                return
            if id not in filtered_document_ids:
                current_position += 1
            else:
                document_data = self.document_store.get_item(id=id, pool=pool)
                document_data["score"] = docs_scores[id]
                doc = TextDocument.from_dict(document_data)

                yield doc

                returned_docs += 1
                current_position += 1

    def _embedding_retrieval(
        self,
        queries: List[Query],
        filters: Dict[str, Any],
        top_k: int,
        pool: str,
        similarity: str,
        batch_size: int,
        scale_score: bool,
    ) -> Dict[str, List[Document]]:
        """
        Performs retrieval by embedding.
        """
        results: Dict[str, List[Document]] = {}
        for query in queries:
            if query is None or not query.content:
                logger.info(
                    "You tried to perform retrieval on an empty query (%s). No documents returned for it.", query
                )
                results[query] = []
                continue
            if query.embedding is None:
                raise MissingEmbeddingError(
                    "You tried to perform retrieval by embedding similarity on a query without embedding (%s). "
                    "Please compute the embeddings for all your queries before using this method.", 
                    query
                )

            filtered_documents = self.document_store.get_items(
                pool=pool, filters=filters
            )
            try:
                ids, embeddings = zip(
                    *[(doc["id"], doc["embedding"]) for doc in filtered_documents]
                )
            except KeyError:
                # FIXME make it nodeable
                raise MissingEmbeddingError(
                    "Some of the documents don't have embeddings. Use the Embedder to compute them."
                )

            # At this stage the iterable gets consumed.
            if self.device and self.device.type == "cuda":
                scores = get_scores_torch(
                    query=query.embedding,
                    documents=embeddings,
                    similarity=similarity,
                    batch_size=batch_size,
                    device=self.device,
                )
            else:
                embeddings = np.array(embeddings)
                scores = get_scores_numpy(
                    query.embedding, embeddings, similarity=similarity
                )

            top_k_ids = sorted(
                list(zip(ids, scores)),
                key=lambda x: x[1] if x[1] is not None else 0.0, reverse=True
            )[:top_k]

            relevant_documents = []
            for id, score in top_k_ids:
                document_data = self.document_store.get_item(id=id, pool=pool)
                if scale_score:
                    score = scale_to_unit_interval(score, similarity)
                document_data["score"] = score
                document = TextDocument.from_dict(dictionary=document_data)
                relevant_documents.append(document)

            results[query.id] = relevant_documents

        return results
