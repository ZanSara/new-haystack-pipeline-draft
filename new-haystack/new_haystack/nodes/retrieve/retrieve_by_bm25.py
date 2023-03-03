from typing import Dict, Any, List, Tuple

import logging

from new_haystack.data import TextQuery
from new_haystack.nodes import haystack_node


@haystack_node
class RetrieveByBM25:
    """
    Simple dummy BM25 Retriever that works with MemoryStore.
    Supports batch processing.
    """
    def __init__(self, 
            input_name: str = "query", 
            output_name: str = "documents_by_query", 
            default_store: str = "documents", 
            default_top_k: int = 10
        ):
        self.default_store = default_store
        self.default_top_k = default_top_k

        # Pipelne's contract:
        self.init_parameters = {
            "input_name": input_name, 
            "output_name": output_name, 
            "default_store": default_store,
            "default_top_k": default_top_k
        }
        self.expected_inputs = [input_name]
        self.expected_outputs = [output_name]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        my_parameters = parameters.get(name, {})
        store_name = my_parameters.pop("store", self.default_store)
        top_k = my_parameters.pop("top_k", self.default_top_k)

        # This can be done safely, because Nodes expect the Pipeline to respect their contract.
        # Errors here are Pipeline's responsibility, so Nodes should not care.
        queries = data[0][1]

        # Batch support is not the pipeline's business, but the node's
        if isinstance(queries, TextQuery):
            queries = [queries]
        elif queries and not (
            isinstance(queries, list) and 
            all(isinstance(query, TextQuery) for query in queries)
        ):
            raise ValueError(f"'{data[0][0]}' can only contain TextQuery objects. '{data[0][0]}' contains: {queries}")
        
        if not store_name in stores.keys():
            raise ValueError(f"No store called '{store_name}'.")

        results = stores[store_name].get_relevant_documents(queries=queries, top_k=top_k)

        return {self.expected_outputs[0]: results}
    