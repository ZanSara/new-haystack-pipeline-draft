from typing import Dict, Any, Callable
import inspect
import logging


logger = logging.getLogger(__name__)

class NodeError(Exception):
    pass


class NodeValidationError(NodeError):
    pass


def relevant_arguments(
    callable: Callable[..., Any],
    name: str,
    data: Dict[str, Any],
    parameters: Dict[str, Dict[str, Any]],
):
    """
    Finds which arguments the simplified node expects. Reads the function signature and
    returns the values corresponding to each parameter name from the data and parameters
    dictionaries.
    """
    signature = inspect.signature(callable)
    if any(
        signature.parameters[name].kind == inspect.Parameter.VAR_POSITIONAL
        for name in signature.parameters
    ):
        raise NodeError(
            "'haystack_simple_node' can only handle functions without *args. Use a list instead."
        )
    if any(
        signature.parameters[name].kind == inspect.Parameter.VAR_KEYWORD
        for name in signature.parameters
    ):
        raise NodeError(
            "'haystack_simple_node' can only handle functions without **kwargs. Use a dictionary instead."
        )

    # Check if there are unexpected parameters
    unexpected_params = {
        key: value
        for key, value in parameters.get(name, {}).items()
        if key not in signature.parameters
    }
    if unexpected_params:
        logger.error(
            "%s received one or more unexpected parameter(s): %s. They will be ignored.",
            name,
            unexpected_params,
        )

    # Filter out what the node expects
    filtered_data = {
        key: value for key, value in data.items() if key in signature.parameters
    }
    filtered_params = {
        key: value
        for key, value in parameters.get(name, {}).items()
        if key in signature.parameters
    }

    node_kwargs = {**filtered_data, **filtered_params}
    logger.debug("%s kwargs: %s", name, node_kwargs)
    return node_kwargs
