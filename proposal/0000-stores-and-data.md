- Title: Stores and Data
- Decision driver: @ZanSara
- Start Date: (today's date, in format YYYY-MM-DD)
- Proposal PR: (fill in after opening the PR)
- Github Issue or Discussion: (only if available, link the original request for this change)

# Summary

Haystack's Document Stores are a very central component in Haystack, and as the name suggest, they were initially designed around the concept of `Document`. 

As the framework grew, so did the number of Document Stores and their API, until the point where keeping them aligned aligned on the same feature set started to become a serious challenge.

In this proposal we outline a reviewed design of the same concept: the `Store`, a class that can do more than just storing documents, by doing less that what `DocumentStore`s do right now :slightly_smiling_face:

Note: these stores are designed to work alongside Haystack 2.0 Pipelines (see https://github.com/deepset-ai/haystack/pull/4284)

# Motivation

Current `DocumentStore` face several issues mostly due to their organic growth. Some of them are:

- Very aware of what they are storing. They are strictly oriented towards storing `Document`s, or `Label`s, and they generally react badly to new primitives or `content_type`s.

- `DocumentStore`s perform the bulk of retrieval, but they are need to be tighly coupled to a `Retriever` object to work. We believe this coupling can be broken by a clear API boundary between `Stores`, `Retriever`s and `Embedder`s (see below)

- `DocumentStore`s tend to bring in complex dependencies, so less used ones should be easy to decouple into external packages at need.

- We have no strategy to store files, which is going to be an increasing concern with the rise of multimodality.

# Basic example

The proposal focuses on two hierarchies: one for the stores and one for the data they store.

## The `Data` hierarchy

We define a Data object that has only two fields: `content`, `meta`, and `id_hash_keys`. From there, we create a matrix of subclasses by "objective" (`Documents`, `Answer`, `Query`, `Label`, etc), and by "content type" (`Text`, `Table`, `Image`, etc).

This double matrix allows for more accurate validation with Pydantic.

## The `Store` hierarchy

Stores follow a similar pattern. We define a baseclass `Store` that defines a simple CRUD API. This API stores generic `Data` objects (or even simple dictionaries, TBD).

By subclassing it we provide one implementation for each underlying technology (`MemoryStore`, `ElastisearchStore`, `FaissStore`). These implementations should be very simple, because they are supposed to perform only basic CRUD, **not retrieval**.

On top of these, we then provide subclasses of each of these specialized stores that handles specific `Data` subclasses. For example, we could have `MemoryDocumentStore`, `MemoryLabelStore`, `MemoryQueryStore`, etc...

In this case, we foresee implementing mostly `DocumentStore` variants, because those will add support for dense/sparse retrieval. This, however, does not prevent the creation of other specialized subclasses if the need shows.

# Detailed design

## Design of the `Data` hierarchy

The design for the Data subclasses is fairly straigthforward and consists mostly of very small, immutable dataclasses. Here we propose an implementation example with `Data`, its content type variants, `Document` and its content types variants again.

```python
from typing import List, Any, Dict, Literal
from math import inf
from pathlib import Path
import logging
import json
from dataclasses import asdict, dataclass, field
import mmh3

#: List of all `content_type` supported
ContentTypes = Literal["text", "table", "image", "audio"]

@dataclass(frozen=True, kw_only=True, eq=True)
class Data:

    id: str = field(default_factory=str)
    content: Any = field(default_factory=lambda: None)
    content_type: ContentTypes
    meta: Dict[str, Any] = field(default_factory=dict, hash=False)
    id_hash_keys: List[str] = field(default_factory=lambda: ["content"], hash=False)

    def __str__(self):
        return f"{self.__class__.__name__}('{self.content}')"

    def to_dict(self):
        return asdict(self)

    def to_json(self, **json_kwargs):
        return json.dumps(self.to_dict(), *json_kwargs)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)

    @classmethod
    def from_json(cls, data, **json_kwargs):
        dictionary = json.loads(data, **json_kwargs)
        return cls.from_dict(dictionary=dictionary)

@dataclass(frozen=True, kw_only=True)
class Text(Data):
    content: str
    content_type: ContentTypes = "text"

@dataclass(frozen=True, kw_only=True)
class Table(Data):
    content: 'pd.DataFrame'
    content_type: ContentTypes = "table"

@dataclass(frozen=True, kw_only=True)
class Image(Data):
    content: Path
    content_type: ContentTypes = "image"

@dataclass(frozen=True, kw_only=True)
class Audio(Data):
    content: Path
    content_type: ContentTypes = "audio"


@dataclass(frozen=True, kw_only=True)
class Document(Data):

    score: Optional[float] = None
    embedding: Optional[np.ndarray] = field(default=lambda:None, repr=False)

    def __lt__(self, other):
        if not hasattr(other, "score"):
            raise ValueError("Documents can only be compared with other Documents.")
        return (self.score if self.score is not None else -inf) < (
            other.score if other.score is not None else -inf
        )

    def __le__(self, other):
        if not hasattr(other, "score"):
            raise ValueError("Documents can only be compared with other Documents.")
        return (self.score if self.score is not None else -inf) <= (
            other.score if other.score is not None else -inf
        )

@dataclass(frozen=True, kw_only=True)
class TextDocument(Text, Document):
    pass

@dataclass(frozen=True, kw_only=True)
class TableDocument(Table, Document):
    pass

@dataclass(frozen=True, kw_only=True)
class ImageDocument(Image, Document):
    pass

@dataclass(frozen=True, kw_only=True)
class AudioDocument(Audio, Document):
    pass
```

## Design of the `Store` hierarchy

`Store`s are more complex as they have to be disentangled from `Retriever`s.

First, let's define the API of a basic `Store`:

```python
class Store(ABC):

    def __init__(self, pool: str):
        pass

    def create_pool(self, pool: str, use_bm25: Optional[bool] = None) -> None:
        pass

    def list_pools(self) -> List[str]:
        pass

    def delete_pool(self, pool: str, delete_populated_pool: bool = False) -> None:
        pass

    def has_item(self, id: str, pool: str) -> bool:
        pass

    def get_item(self, id: str, pool: str) -> Data:
        pass

    def count_items(self, filters: Dict[str, Any], pool: str) -> int:
        pass

    def get_ids(self, filters: Dict[str, Any], pool: str) -> Iterable[str]:
        pass

    def get_items(
        self, filters: Dict[str, Any], pool: str
    ) -> Iterable[Data]:
        pass

    def write_items(
        self,
        items: Iterable[Data],
        pool: str,
        duplicates: Literal["skip", "overwrite", "fail"],
    ) -> None:
        pass

    def delete_items(
        self, ids: List[str], pool: str, fail_on_missing_item: bool = False
    ) -> None:
        pass
```

As you can see, most concepts were kept from old DocumentStores, with a few notable exceptions:

- `index` became `pool`, to take some distance from Elasticsearch `index`, which causes noticeable confusion to FAISS users due to a naming collision.

- There's no more `count_documents`, `get_documents`, etc, but only `count_items`, `get_item`. This store is generic and can contain any type of data. Therefore, we can separate storing Documents from storing Labels and we don't need to care about supporting both: if a generic Store exists for that backend, it can automatically store both Documents and Labels with trivial implementation effort.

- No mention of retrieval other than `get_items`, which only accepts `filters`. That's because `get_items` is NOT supposed to be used for retrieval, but only, if needed, for filtering.

Let's now assume we create a `MemoryStore` that simply implements the methods above. From such class we could then create a specialized subclass called `MemoryDocumentStore`, which would look like this:

```python
class MemoryDocumentStore:

    def __init__(self, ...params...):
        self.store = MemoryStore(pool="documents")
        ...

    def create_pool(self, pool: str, use_bm25: Optional[bool] = None):
        self.store.create_pool(pool=pool)

        # Additional code to support BM25 retrieval
        if use_bm25 or (use_bm25 is None and self.use_bm25):
            self.bm25[pool] = BM25Representation(
                bm25_algorithm=self.bm25_algorithm,
                bm25_parameters=self.bm25_parameters,
                bm25_tokenization_regex=self.bm25_tokenization_regex,
            )

    def list_pools(self) -> List[str]:
        ...

    def delete_pool(
        self, pool: str = "documents", delete_populated_pool: bool = False
    ) -> None:
        self.store.delete_pool(
            pool=pool, delete_populated_pool=delete_populated_pool
        )
        # Additional code to support BM25 retrieval
        if self.bm25:
            del self.bm25[pool]

    def has_document(self, id: str, pool: str = "documents") -> bool:
        ...

    def get_document(self, id: str, pool: str = "documents") -> Optional[Document]:
        ...

    def count_documents(self, filters: Dict[str, Any], pool: str = "documents"):
        ...

    def get_document_ids(
        self, filters: Dict[str, Any], pool: str = "documents"
    ) -> Iterable[str]:
        ...

    def get_documents(
        self, filters: Dict[str, Any], pool: str = "documents"
    ) -> Iterable[Document]:
        ...

    def write_documents(
        self,
        documents: Iterable[Document],
        pool: str = "documents",
        duplicates: Literal["skip", "overwrite", "fail"] = "overwrite",
    ) -> None:
        self.store.write_items(
            items=(doc.to_dict() for doc in documents),
            pool=pool,
            duplicates=duplicates,
        )
        # Additional code to support BM25 retrieval
        if self.bm25:
            self.bm25[pool].update_bm25(self.get_documents(filters={}, pool=pool))

    def delete_documents(
        self,
        ids: List[str],
        pool: str = "documents",
        fail_on_missing_item: bool = False,
    ) -> None:
        ...

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

        ######################
        # Performs retrieval #
        ######################

        filters = filters or {}

        # BM25 Retrieval
        if use_bm25:
            relevant_documents = {}
            for query in queries:
                relevant_documents[query] = list(self._bm25_retrieval(...))
            return relevant_documents

        # Embedding Retrieval
        relevant_documents = self._embedding_retrieval(...)
        return relevant_documents

    def _bm25_retrieval(...) -> List[Document]:
        ...

    def _embedding_retrieval(...) -> Dict[str, List[Document]]:
        ...
```

Note two important details:

- `MemoryDocumentStore` DOES NOT inherit from `MemoryStore`. It uses the store internally. This spares us the "signature creep" seen in many DocStores currently, which are being forced by MyPy to have all identical signatures even in such context where it makes no sense (how all DocStores have to support `headers` and then just throw warnings if set). Composition in this case is an extremely valuable tool.

- On the other hand, `MemoryDocumentStore` only adds one single method to the `Store` signature: `get_relevant_documents`. This method is part of the `Retriever`s contract (see below), and that's not enforced by any inheritance, thus leaving the the signature and arguments lists quite free.

## The `Retriever`s contract

We define an explicit contract between `DocumentStore`s (not all `Store`s, just their `Document` variety) and `Retriever`s. This contract is very simple and states that:

> All document stores that support retrieval should define a get_relevant_documents() method.

`Retriever`s are then going to try-catch this method, because it might have a signature they don;t expect. Each `Retriever` will define a set of parameters it is going to pass to its `DocumentStore`, and all `DocumentStore`s that canwant to support that `Retriever` type must accept its set of parameters.

For example:

```python
@node
class RetrieverType1:

    def run(...):
        ...
        if not hasattr(stores[store_name], "get_relevant_documents"):
            raise ValueError(f"{store_name} is not a DocumentStore or it does not support Retrievers.")

        try:
            documents = stores[store_name].get_relevant_documents(use_bm25=True, top_k=10)
        except Exception as e:
            # Note: we might also actively check the signature instead of try-catching
            # This is just an example implementation
            raise ValueError(f"{store_name} is not a DocumentStore type compatible with 'RetrieverType1'.")
        ...
```

# Drawbacks


# Alternatives


# Adoption strategy


# How we teach this


# Unresolved questions
