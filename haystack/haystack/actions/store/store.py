from typing import Dict, List, Any

import logging

from haystack.actions import haystack_action, ActionError


logger = logging.getLogger(__name__)


@haystack_action
def store_documents(
    name: str,
    data: Dict[str, Any],
    parameters: Dict[str, Any],
    outgoing_edges: List[str],
    stores: Dict[str, Any],
):
    input_variable_name = parameters.get("documents", "documents")
    store_name = parameters.get("store_name", "documents")
    index = parameters.get("index", "documents")
    duplicates = parameters.get("duplicates", "overwrite")

    # FIXME Fail or log?
    if not input_variable_name in data.keys():
        logger.error(
            "No documents to store! '%s' is not present in the pipeline context. "
            "'%s' won't write anything in the document store.",
            input_variable_name,
            name,
        )

    # FIXME Fail or log?
    if not store_name in stores.keys():
        raise ActionError(f"No store called {store_name}.")
    if not hasattr(stores[store_name], "write_documents"):
        raise ActionError(f"The store called {store_name} is not a DocumentStore.")

    stores[store_name].write_documents(
        documents=data[input_variable_name], index=index, duplicates=duplicates
    )

    return {edge: (data, parameters) for edge in outgoing_edges}
