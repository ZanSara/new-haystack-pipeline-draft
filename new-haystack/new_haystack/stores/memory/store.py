from typing import Literal, Any, Dict, List, Optional, Iterable

import logging

from new_haystack.stores._utils import (
    DuplicateError,
    IndexFullError,
    MissingItemError,
    MissingIndexError,
)


logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Stores data in-memory, without persisting it anywhere. 
    Ephemeral storage that can't (yet) be saved to disk in any way.
    """

    def __init__(
        self,
        index: str,
    ):
        self.indexes: Dict[str, Dict[str, Any]] = {index: {}}

    def create_index(self, index: str, use_bm25: Optional[bool] = None) -> None:
        """
        Creates a new index with the given name.

        :param index: the index name
        """
        self.indexes[index] = {}

    def list_indexes(self) -> List[str]:
        """
        Returns a list of all the indexes present in this store.
        """
        return list(self.indexes.keys())

    def delete_index(self, index: str, delete_populated_index: bool = False) -> None:
        """
        Drops an index completely. Will not delete index that contains items unless
        `delete_populated_index=True` (default is False).

        :param index: the index to drop
        :param delete_populated_index: whether to drop full indexes too
        :raises IndexFullError if the index is full and delete_populated_index=False
        """
        items_count = self.count_items(filters={}, index=index)
        if items_count > 0:
            if not delete_populated_index:
                raise IndexFullError(
                    f"You tried to delete index '{index}' which contains {items_count} entries. "
                    "Use `delete_populated_index=True` if you're sure."
                )
            logger.warning(
                f"You are deleting index '{index}' which contains {items_count} items."
            )
        del self.indexes[index]

    def has_item(self, id: str, index: str) -> bool:
        """
        Checks if this ID exists in the store.

        :param id: the id to find in the store.
        :param index: in which index to look for this item.
        """
        try:
            return id in self.indexes[index].keys()
        except IndexError as e:
            raise MissingIndexError(
                f"No index names {index}. Create it with .create_index()"
            ) from e

    def get_item(self, id: str, index: str) -> Dict[str, Any]:
        """
        Finds a item by ID in the store. Fails if the item is not present.

        Not to be used for retrieval or filtering.

        :param id: the id of the item to get.
        :param index: in which index to look for this item.
        """
        try:
            if not self.has_item(id=id, index=index):
                raise MissingItemError(f"ID {id} not found in index {index}.")
            return self.indexes[index][id]
        except IndexError as e:
            raise MissingIndexError(
                f"No index names {index}. Create it with .create_index()"
            ) from e

    def count_items(self, filters: Dict[str, Any], index: str) -> int:
        """
        Returns the number of how many items match the given filters.
        Pass filters={} to count all items in the given index.

        :param filters: the filters to apply to the items list.
        :param index: in which index to look for this item.
        """
        return len(self.indexes[index].keys())

    def get_ids(self, filters: Dict[str, Any], index: str) -> Iterable[str]:
        """
        Returns only the IDs of the items that match the filters provided.

        :param filters: the filters to apply to the item list.
        :param index: in which index to look for this item.
        """
        try:
            # TODO apply filters
            for id in self.indexes[index].keys():
                yield id
        except IndexError as e:
            raise MissingIndexError(
                f"No index names {index}. Create it with .create_index()"
            ) from e

    def get_items(
        self, filters: Dict[str, Any], index: str
    ) -> Iterable[Dict[str, Any]]:
        """
        Returns the items that match the filters provided.

        :param filters: the filters to apply to the item list.
        :param index: in which index to look for this item.
        """
        for id in self.get_ids(filters=filters, index=index):
            yield self.indexes[index][id]

    def write_items(
        self,
        items: Iterable[Dict[str, Any]],
        index: str,
        duplicates: Literal["skip", "overwrite", "fail"],
    ) -> None:
        """
        Writes items into the store.

        :param items: a list of dictionaries.
        :param index: the index to write items into
        :param duplicates: items with the same ID count as duplicates. When duplicates are met,
            Haystack can choose to:
             - skip: keep the existing item and ignore the new one.
             - overwrite: remove the old item and write the new one.
             - fail: an error is raised
        :raises DuplicateError: Exception trigger on duplicate item
        :return: None
        """
        for item in items:
            if self.has_item(item["id"], index=index):
                if duplicates == "fail":
                    raise DuplicateError(
                        f"ID {item['id']} already exists in index '{index}'."
                    )
                elif duplicates == "skip":
                    logger.warning(
                        "ID '%s' already exists in index '%s'", item["id"], index
                    )
            self.indexes[index][item["id"]] = item

    def delete_items(
        self, ids: List[str], index: str, fail_on_missing_item: bool = False
    ) -> None:
        """
        Deletes all ids from the given index.

        :param ids: the ids to delete
        :param index: the index where these id should be stored
        :param fail_on_missing_item: fail if the id is not found, log ignore otherwise
        """
        for id in ids:
            if not self.has_item(id=id, index=index):
                if fail_on_missing_item:
                    raise MissingItemError(
                        f"ID {id} not found in index {index}, cannot delete it."
                    )
                logger.info(f"ID {id} not found in index {index}, cannot delete it.")
            try:
                del self.indexes[index][id]
            except IndexError as e:
                raise MissingIndexError(
                    f"No index names {index}. Create it with .create_index()"
                ) from e
