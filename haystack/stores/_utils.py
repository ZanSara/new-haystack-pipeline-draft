
class StoreError:
    pass

class DuplicateError(StoreError):
    pass

class MissingItemError(StoreError):
    pass

class MissingEmbeddingError(StoreError):
    pass

class IndexFullError(StoreError):
    pass
