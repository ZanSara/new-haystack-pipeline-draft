from typing import Literal, Any, Optional, Dict, Callable

from math import inf
from pathlib import Path
import logging
import json
from dataclasses import asdict,  dataclass, field

import mmh3
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


#: List of all `content_type` supported
ContentTypes = Literal["text", "table", "image", "audio"]


DEFAULT_ID_HASH_FUNCTION = lambda doc: "{:02x}".format(mmh3.hash128(str(doc.content), signed=False))


@dataclass(frozen=True, kw_only=True)
class Document:
    meta: Dict[str, Any] = field(default_factory=dict)
    id_hash_function: Callable[['Document'], str] = field(default=DEFAULT_ID_HASH_FUNCTION, repr=False)
    score: Optional[float] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        object.__setattr__(self, 'id', self.id_hash_function(self))
    
    def __lt__(self, other):
        if not hasattr(other, "score"):
            raise ValueError("Documents can only be compared with other Documents.")
        return (self.score if self.score is not None else -inf) < (other.score if other.score is not None else -inf)

    def __le__(self, other):
        if not hasattr(other, "score"):
            raise ValueError("Documents can only be compared with other Documents.")
        return (self.score if self.score is not None else -inf) <= (other.score if other.score is not None else -inf)

    def to_dict(self):
        return asdict(self, dict_factory=lambda x: {k: v for (k, v) in x if k != "id_hash_function"})

    def to_json(self, indent: int = 0, default: Callable[..., Any] = str):
        return json.dumps(self.to_dict(), indent=indent, default=default)

    @classmethod
    def from_dict(cls, dictionary, id_hash_function: Callable[['Document'], str] = DEFAULT_ID_HASH_FUNCTION):
        return cls(**dictionary, id_hash_function=id_hash_function)

    @classmethod
    def from_json(cls, data, id_hash_function: Callable[['Document'], str] = DEFAULT_ID_HASH_FUNCTION):
        dictionary = json.loads(data)
        return cls.from_dict(dictionary=dictionary, id_hash_function=id_hash_function)


@dataclass(frozen=True, kw_only=True)
class TextDocument(Document):
    content: str
    content_type: ContentTypes = "text"


@dataclass(frozen=True, kw_only=True)
class TableDocument(Document):
    content: pd.DataFrame
    content_type: ContentTypes = "table"


@dataclass(frozen=True, kw_only=True)
class ImageDocument(Document):
    content: Path
    content_type: ContentTypes = "image"


@dataclass(frozen=True, kw_only=True)
class AudioDocument(Document):
    content: Path
    content_type: ContentTypes = "audio"
