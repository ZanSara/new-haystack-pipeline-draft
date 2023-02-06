from typing import Literal, Any, Dict, List, Optional, Union, Generator

import time
import logging
from copy import deepcopy
from collections import defaultdict
import re

import numpy as np
from tqdm.auto import tqdm

from haystack.data import Document
from haystack.models.device_management import initialize_device_settings
from haystack.document_stores._utils import DocumentStoreError, DuplicateDocumentError


logger = logging.getLogger(__name__)


try:
    import torch
except ImportError as e:
    logger.debug("torch not found: MemoryDocumentStore won't be able to search by embedding with a local model.")

try:
    import rank_bm25
except ImportError as e:
    logger.debug("rank_bm25 not found: MemoryDocumentStore won't be able to build its BM25 index.")



class MemoryDocumentStore:

    def __init__(
        self,
        use_bm25: bool = True,
        bm25_tokenization_regex: str = r"(?u)\b\w\w+\b",
        bm25_algorithm: Literal["BM25Okapi", "BM25L", "BM25Plus"] = "BM25Okapi",
        bm25_parameters: dict = {},
        use_gpu: bool = True,
        devices: Optional[List[Union[str, 'torch.device']]] = None,
        progress_bar: bool = True,
    ):
        self.indexes: Dict[str, Dict] = {"documents": {}}
        self.progress_bar = progress_bar

        self.bm25 = {}
        if use_bm25:
            self.bm25 = {
                "documents": BM25Representation(
                    bm25_tokenization_regex=bm25_tokenization_regex,
                    bm25_algorithm=bm25_algorithm,
                    bm25_parameters=bm25_parameters
                )
            }

        devices, _ = initialize_device_settings(devices=devices, use_cuda=use_gpu, multi_gpu=False)
        self.device = None
        if len(devices) > 1:
            logger.warning(
                "Multiple devices are not supported in %s inference, using the first device %s.",
                self.__class__.__name__,
                devices[0],
            )
            self.device = devices[0]
    
    def create_index(self):
        pass

    def list_indexes(self):
        pass

    def delete_index(self):
        pass

    def has_document(self, id: str, index: str = "documents") -> bool:
        """
        Checks if this ID exists in the document store.

        :param id: the id to find in the document store.
        :param index: in which index to look for this document.
        """
        try:
            return id in self.indexes[index].keys()
        except IndexError as e:
            raise DocumentStoreError(f"No index names {index}. Create it with .create_index()") from e

    def get_document(self, id: str, index: str = "documents") -> Optional[Document]:
        """
        Finds a document by ID in the document store. Returns None if the document is not present.

        Not to be used for retrieval or filtering, use .get_similar_documents() and .get_filtered_documents().

        :param id: the id of the document to get.
        :param index: in which index to look for this document.
        """
        try:
            return self.indexes[index].get(id, None)
        except IndexError as e:
            raise DocumentStoreError(f"No index names {index}. Create it with .create_index()") from e

    def get_filtered_documents(self, filters: Dict[str, Any], index: str = "documents") -> Generator[Document, None, None]:
        """
        Filters the content of the document store using the filters provided.

        :param filters: the filters to apply to the documents list.
        :param index: in which index to look for this document.
        """
        try:
            for doc in self.indexes[index].values():
                yield doc
        except IndexError as e:
            raise DocumentStoreError(f"No index names {index}. Create it with .create_index()") from e
    
    def get_similar_documents(self, ids: Iterator[str], index: str = "documents") -> Generator[List[Document], None, None]:
        """
        Retrieves the documents by embedding similarity.

        :param ids: the ids of the documents to search for.
        :param index: in which index to look for this document.
        """
        pass
        # try:
        #     for doc in self.indexes[index].values():
        #         yield doc
        # except IndexError as e:
        #     raise DocumentStoreError(f"No index names {index}. Create it with .create_index()") from e

    def write_documents(
        self,
        documents: List[Document],
        index: str = "documents",
        duplicates: Literal["skip", "overwrite", "fail"] = "overwrite",
    ):
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
        is_duplicate = [self.has_document(doc.id) for doc in documents]
        if duplicates == "fail":
            duplicate_docs = [documents[pos] for pos in is_duplicate]
            raise DuplicateDocumentError(
                f"Document with ids {duplicate_docs} already exists in index '{index}'."
            )
        elif duplicates == "skip":
            # Emitting on purpose one warning for each document: it has to be loud.
            for pos in is_duplicate:
                logger.warning(
                    "Document with id '%s' already exists in index '%s'", documents[pos].id, index
                )
            documents = [doc for pos, doc in enumerate(documents) if pos not in is_duplicate]

        # Update BM25 index
        if self.bm25 and len(documents) > 0:
            self.bm25[index].update_bm25(self.get_filtered_documents(filters={}))

        # Write the docs
        self.indexes[index].update({doc.id: doc for doc in documents})

    def delete_documents(self):
        pass

    def get_similar_documents(self):
        pass





class BM25Representation:

    def __init__(
        self,
        bm25_tokenization_regex: str = r"(?u)\b\w\w+\b",
        bm25_algorithm: Literal["BM25Okapi", "BM25L", "BM25Plus"] = "BM25Okapi",
        bm25_parameters: dict = {},
    ):
        self.bm25_tokenization_regex = bm25_tokenization_regex
        self.bm25_algorithm = bm25_algorithm
        self.bm25_parameters = bm25_parameters
        self.bm25: Dict[str, rank_bm25.BM25] = {}

    @property
    def bm25_tokenization_regex(self):
        return self._tokenizer

    @bm25_tokenization_regex.setter
    def bm25_tokenization_regex(self, regex_string: str):
        self._tokenizer = re.compile(regex_string).findall

    @property
    def bm25_algorithm(self):
        return self._bm25_class

    @bm25_algorithm.setter
    def bm25_algorithm(self, algorithm: str):
        self._bm25_class = getattr(rank_bm25, algorithm)
    
    def update_bm25(self, documents: Generator[Document, None, None]) -> None:
        """
        Updates the BM25 sparse representation in the the document store.

        :param documents: a generator returning all the documents in the docstore
        """
        tokenized_corpus = []
        for doc in tqdm(documents, unit=" docs", desc="Updating BM25 representation..."):
            if doc.content_type != "text":
                logger.warning(
                    "Document %s is non-textual. It won't be present in the BM25 index.",
                    doc.id,
                )
            else:
                tokenized_corpus.append(self.bm25_tokenization_regex(doc.content.lower()))

        self.bm25 = self.bm25_algorithm(tokenized_corpus, **self.bm25_parameters)
