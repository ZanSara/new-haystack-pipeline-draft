from typing import Literal, Any, Dict, List, Optional, Iterable

import logging

from new_haystack.stores._utils import (
    DuplicateError,
    PoolFullError,
    MissingItemError,
    MissingPoolError,
)


logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Stores data in-memory, without persisting it anywhere. 
    Ephemeral storage that can't (yet) be saved to disk in any way.
    """

    def __init__(
        self,
        pool: str,
    ):
        self.pools: Dict[str, Dict[str, Any]] = {pool: {}}

    def create_pool(self, pool: str, use_bm25: Optional[bool] = None) -> None:
        """
        Creates a new pool with the given name.

        :param pool: the pool name
        """
        self.pools[pool] = {}

    def list_pools(self) -> List[str]:
        """
        Returns a list of all the pools present in this store.
        """
        return list(self.pools.keys())

    def delete_pool(self, pool: str, delete_populated_pool: bool = False) -> None:
        """
        Drops an pool completely. Will not delete pool that contains items unless
        `delete_populated_pool=True` (default is False).

        :param pool: the pool to drop
        :param delete_populated_pool: whether to drop full pools too
        :raises IndexFullError if the pool is full and delete_populated_pool=False
        """
        items_count = self.count_items(filters={}, pool=pool)
        if items_count > 0:
            if not delete_populated_pool:
                raise PoolFullError(
                    f"You tried to delete pool '{pool}' which contains {items_count} entries. "
                    "Use `delete_populated_pool=True` if you're sure."
                )
            logger.warning(
                f"You are deleting pool '{pool}' which contains {items_count} items."
            )
        del self.pools[pool]

    def has_item(self, id: str, pool: str) -> bool:
        """
        Checks if this ID exists in the store.

        :param id: the id to find in the store.
        :param pool: in which pool to look for this item.
        """
        try:
            return id in self.pools[pool].keys()
        except IndexError as e:
            raise MissingPoolError(
                f"No pool names {pool}. Create it with .create_pool()"
            ) from e

    def get_item(self, id: str, pool: str) -> Dict[str, Any]:
        """
        Finds a item by ID in the store. Fails if the item is not present.

        Not to be used for retrieval or filtering.

        :param id: the id of the item to get.
        :param pool: in which pool to look for this item.
        """
        try:
            if not self.has_item(id=id, pool=pool):
                raise MissingItemError(f"ID {id} not found in pool {pool}.")
            return self.pools[pool][id]
        except IndexError as e:
            raise MissingPoolError(
                f"No pool names {pool}. Create it with .create_pool()"
            ) from e

    def count_items(self, filters: Dict[str, Any], pool: str) -> int:
        """
        Returns the number of how many items match the given filters.
        Pass filters={} to count all items in the given pool.

        :param filters: the filters to apply to the items list.
        :param pool: in which pool to look for this item.
        """
        return len(self.pools[pool].keys())

    def get_ids(self, filters: Dict[str, Any], pool: str) -> Iterable[str]:
        """
        Returns only the IDs of the items that match the filters provided.

        :param filters: the filters to apply to the item list.
        :param pool: in which pool to look for this item.
        """
        try:
            # TODO apply filters
            for id in self.pools[pool].keys():
                yield id
        except IndexError as e:
            raise MissingPoolError(
                f"No pool names {pool}. Create it with .create_pool()"
            ) from e

    def get_items(
        self, filters: Dict[str, Any], pool: str
    ) -> Iterable[Dict[str, Any]]:
        """
        Returns the items that match the filters provided.

        :param filters: the filters to apply to the item list.
        :param pool: in which pool to look for this item.
        """
        for id in self.get_ids(filters=filters, pool=pool):
            yield self.pools[pool][id]

    def write_items(
        self,
        items: Iterable[Dict[str, Any]],
        pool: str,
        duplicates: Literal["skip", "overwrite", "fail"],
    ) -> None:
        """
        Writes items into the store.

        :param items: a list of dictionaries.
        :param pool: the pool to write items into
        :param duplicates: items with the same ID count as duplicates. When duplicates are met,
            Haystack can choose to:
             - skip: keep the existing item and ignore the new one.
             - overwrite: remove the old item and write the new one.
             - fail: an error is raised
        :raises DuplicateError: Exception trigger on duplicate item
        :return: None
        """
        for item in items:
            if self.has_item(item["id"], pool=pool):
                if duplicates == "fail":
                    raise DuplicateError(
                        f"ID {item['id']} already exists in pool '{pool}'."
                    )
                elif duplicates == "skip":
                    logger.warning(
                        "ID '%s' already exists in pool '%s'", item["id"], pool
                    )
            self.pools[pool][item["id"]] = item

    def delete_items(
        self, ids: List[str], pool: str, fail_on_missing_item: bool = False
    ) -> None:
        """
        Deletes all ids from the given pool.

        :param ids: the ids to delete
        :param pool: the pool where these id should be stored
        :param fail_on_missing_item: fail if the id is not found, log ignore otherwise
        """
        for id in ids:
            if not self.has_item(id=id, pool=pool):
                if fail_on_missing_item:
                    raise MissingItemError(
                        f"ID {id} not found in pool {pool}, cannot delete it."
                    )
                logger.info(f"ID {id} not found in pool {pool}, cannot delete it.")
            try:
                del self.pools[pool][id]
            except IndexError as e:
                raise MissingPoolError(
                    f"No pool names {pool}. Create it with .create_pool()"
                ) from e
