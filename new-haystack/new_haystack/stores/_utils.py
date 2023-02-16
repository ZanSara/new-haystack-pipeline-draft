class StoreError(Exception):
    pass


class DuplicateError(StoreError):
    pass


class MissingPoolError(StoreError):
    pass


class MissingItemError(StoreError):
    pass


class MissingEmbeddingError(StoreError):
    pass


class PoolFullError(StoreError):
    pass
