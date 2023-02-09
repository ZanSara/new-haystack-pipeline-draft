from typing import Optional

import logging
from dataclasses import dataclass

import numpy as np

from new_haystack.data.data import Data, TextData, TableData, ImageData, AudioData


logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Query(Data):
    """
    Base data class containing a query and in some cases an embedding.

    Can contain text snippets, tables, file paths to images and audio files.
    Please use the subclasses for proper typing.

    Queries can be serialized to/from dictionary and JSON and are immutable.

    id_hash_keys are referring to keys in the meta.
    """
    embedding: Optional[np.ndarray] = None


@dataclass(frozen=True, kw_only=True)
class TextQuery(TextData):
    pass

@dataclass(frozen=True, kw_only=True)
class TableQuery(TableData):
    pass

@dataclass(frozen=True, kw_only=True)
class ImageQuery(ImageData):
    pass

@dataclass(frozen=True, kw_only=True)
class AudioQuery(AudioData):
    pass
