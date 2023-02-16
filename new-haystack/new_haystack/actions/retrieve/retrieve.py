from typing import Dict, List, Any

import logging

from new_haystack.actions import haystack_node, ActionError
from new_haystack.data import TextDocument, Query


logger = logging.getLogger(__name__)


class RetrieverError(ActionError):
    pass


#@haystack_node
def retrieve_by_embedding_similarity(
    name: str,
    data: Dict[str, Any],
    parameters: Dict[str, Any],
    outgoing_edges: List[str],
    stores: Dict[str, Any],
):
    relevant_parameters = parameters.get(name, {})
    query_variable = relevant_parameters.pop("input", "queries")
    store_name = relevant_parameters.pop("store", "documents")
    output_variable = relevant_parameters.pop("output", "relevant_documents")
    unwrap_results = relevant_parameters.pop("unwrap_results", True)
    use_bm25 = relevant_parameters.pop("use_bm25", True)

    # FIXME Fail or log?
    if not query_variable in data.keys():
        raise RetrieverError(
            "Query not found! It was supposed to be in the %s key.",
            query_variable,
        )

    # We decide it's polite for a node to pop its input, so that less data
    # keeps flowing down the pipeline.
    queries = data.pop(query_variable)
    if isinstance(queries, Query):
        queries = [queries]
    elif queries and not (isinstance(queries, list) and all(isinstance(query, Query) for query in queries)):
        raise RetrieverError(f"The variable '{query_variable}' can only contain Query objects! '{query_variable}' contains: {queries}")

    # FIXME Fail or log?
    if not store_name in stores.keys():
        raise ActionError(f"No store called {store_name}.")
    if not hasattr(stores[store_name], "write_documents"):
        raise ActionError(f"The store called {store_name} is not a DocumentStore.")

    documents = stores[store_name].get_relevant_documents(
        queries=queries,
        **relevant_parameters,
        use_bm25=use_bm25,
    )
    if unwrap_results:
        results = [(query, list(documents[query.id])) for query in queries]
    else:
        results = [(query, documents[query.id]) for query in queries]

    data[output_variable] = results

    # TBD: output on all edges the same thing, or enforce everyone to connect
    # to the same "default" edge? For now using strategy one.
    return {edge: (data, parameters) for edge in outgoing_edges}
