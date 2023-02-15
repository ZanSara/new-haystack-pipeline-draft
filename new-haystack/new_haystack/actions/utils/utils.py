from typing import Dict, List, Any

import logging

from new_haystack.data import TextQuery
from new_haystack.actions import haystack_node, ActionError


logger = logging.getLogger(__name__)


@haystack_node
def strings_to_text_queries(
    name: str,
    data: Dict[str, Any],
    parameters: Dict[str, Any],
    outgoing_edges: List[str],
    stores: Dict[str, Any],
):
    input_variable_name = parameters.get(name, {}).get("input", "strings")
    output_variable_name = parameters.get(name, {}).get("output", "queries")
    query_constructor = parameters.get(name, {}).get("query_constructor_params", {})
    
    # FIXME Fail or log?
    if not input_variable_name in data.keys():
        logger.error(
            "No strings to convert! '%s' is not present in the pipeline context. "
            "'%s' won't create any Query objects on '%s'.",
            input_variable_name,
            name,
            output_variable_name
        )

    queries = []
    for string in data.pop(input_variable_name, []):
        queries.append(TextQuery(content=string, **query_constructor))
    data[output_variable_name] = queries
    
    return {edge: (data, parameters) for edge in outgoing_edges}
