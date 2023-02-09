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
        self.document_store = MemoryStore(index="documents")

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

    def create_index(self, index: str, use_bm25: Optional[bool] = None):
        """
        Creates a new index with the given name.
        The BM25 representation is initialized according to use_bm25 if given,
        otherwise defaults to the value given at init time.

        :param index: the index name
        :param use_bm25: whether to initialize the BM25 representation for this index
        """
        self.document_store.create_index(index=index)

        if use_bm25 or (use_bm25 is None and self.use_bm25):
            self.bm25[index] = BM25Representation(
                bm25_algorithm=self.bm25_algorithm,
                bm25_parameters=self.bm25_parameters,
                bm25_tokenization_regex=self.bm25_tokenization_regex,
            )

    def list_indexes(self) -> List[str]:
        """
        Returns a list of all the indexes present in this store.
        """
        return self.list_indexes()

    def delete_index(
        self, index: str = "documents", delete_populated_index: bool = False
    ) -> None:
        """
        Drops an index completely. Will not delete index that contains items unless
        `delete_populated_index=True` (default is False).

        :param index: the index to drop
        :param delete_populated_index: whether to drop full indexes too
        :raises IndexFullError if the index is full and delete_populated_index=False
        """
        self.document_store.delete_index(
            index=index, delete_populated_index=delete_populated_index
        )
        if self.bm25:
            del self.bm25[index]

    def has_document(self, id: str, index: str = "documents") -> bool:
        """
        Checks if this ID exists in the document store.

        :param id: the id to find in the document store.
        :param index: in which index to look for this document.
        """
        return self.document_store.has_item(id=id, index=index)

    def get_document(self, id: str, index: str = "documents") -> Optional[Document]:
        """
        Finds a document by ID in the document store.

        :param id: the id of the document to get.
        :param index: in which index to look for this document.
        """
        return Document.from_dict(self.document_store.get_item(id=id, index=index))

    def count_documents(self, filters: Dict[str, Any], index: str = "documents"):
        """
        Returns the number of how many documents match the given filters.
        Pass filters={} to count all documents in the given index.

        :param filters: the filters to apply to the documents list.
        :param index: in which index to look for this document.
        """
        return self.document_store.count_items(filters=filters, index=index)

    def get_document_ids(
        self, filters: Dict[str, Any], index: str = "documents"
    ) -> Iterable[str]:
        """
        Returns only the ID of the documents that match the filters provided.

        :param filters: the filters to apply to the documents list.
        :param index: in which index to look for this document.
        """
        return self.document_store.get_ids(filters=filters, index=index)

    def get_documents(
        self, filters: Dict[str, Any], index: str = "documents"
    ) -> Iterable[Document]:
        """
        Returns the documents that match the filters provided.

        :param filters: the filters to apply to the documents list.
        :param index: in which index to look for this document.
        """
        for doc in self.document_store.get_items(filters=filters, index=index):
            yield Document.from_dict(dictionary=doc)

    def write_documents(
        self,
        documents: Iterable[Document],
        index: str = "documents",
        duplicates: Literal["skip", "overwrite", "fail"] = "overwrite",
    ) -> None:
        """
        Writes documents into the store.

        :param documents: a list of Haystack Document objects.
        :param index: write documents to a different namespace. The default namespace is 'documents'.
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
            index=index,
            duplicates=duplicates,
        )
        if self.bm25 and len(documents) > 0:
            self.bm25[index].update_bm25(self.get_documents(filters={}, index=index))

    def delete_documents(
        self,
        ids: List[str],
        index: str = "documents",
        fail_on_missing_item: bool = False,
    ) -> None:
        """
        Deletes all ids from the given index.

        :param ids: the ids to delete
        :param index: the index where these id should be stored
        :param fail_on_missing_item: fail if the id is not found, log ignore otherwise
        """
        self.document_store.delete_items(
            ids=ids, index=index, fail_on_missing_item=fail_on_missing_item
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
        index: str = "documents",
    ) -> Dict[str, List[Document]]:
        """
        Performs document retrieval, either by BM25, or by embedding, according to the input parameters.

        :param queries: either strings for bm25 retrieval or embeddings for embedding retrieval.
        :param filters: return only documents that match these filters too
        :param top_k: how many documents to return at most. Might return less documents if the filters
            don't return enough documents.
        :param use_bm25: whether to do the retrieval with bm25 or embeddings
        :param index: the index to get the documents from.
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
                    index=index,
                )
            return relevant_documents

        # Embedding Retrieval
        relevant_documents = self._embedding_retrieval(
            queries=queries,
            filters=filters,
            top_k=top_k,
            index=index,
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
        index: str,
    ) -> List[Document]:
        """
        Performs BM25 retrieval using the rank_bm25 indexes.
        """
        if query is None or not query.content:
            logger.info(
                "You tried to perform retrieval on an empty query. No documents returned for it."
            )
            return []

        if index not in self.bm25.keys():
            raise BM25RepresentationMissing(
                f"No BM25 representation for index {index}"
            )  # TODO add a way to create such index

        filtered_document_ids = (
            self.get_document_ids(
                filters={**filters, "content_type": "text"}, index=index
            ),
        )
        tokenized_query = self.bm25[index].bm25_tokenization_regex(query.content.lower())
        docs_scores = self.bm25[index].bm25.get_scores(tokenized_query)
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
                document_data = self.document_store.get_item(id=id, index=index)
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
        index: str,
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
                index=index, filters=filters
            )
            try:
                ids, embeddings = zip(
                    *[(doc["id"], doc["embedding"]) for doc in filtered_documents]
                )
            except KeyError:
                # FIXME make it actionable
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
                document_data = self.document_store.get_item(id=id, index=index)
                if scale_score:
                    score = scale_to_unit_interval(score, similarity)
                document_data["score"] = score
                document = TextDocument.from_dict(dictionary=document_data)
                relevant_documents.append(document)

            results[query.id] = relevant_documents

        return results
