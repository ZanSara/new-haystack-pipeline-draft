from typing import Dict, Any, Callable
import inspect
import logging


logger = logging.getLogger(__name__)


DEFAULT_EDGE_NAME = "all"


class ActionError(Exception):
    pass

class ActionValidationError(ActionError):
    pass


def relevant_arguments(
    callable: Callable[..., Any], name: str, data: Dict[str, Any], parameters: Dict[str, Dict[str, Any]]
):
    """
    Finds which arguments the simplified action expects. Reads the function signature and
    returns the values corresponding to each parameter name from the data and parameters
    dictionaries.
    """
    signature = inspect.signature(callable)
    if any(signature.parameters[name].kind == inspect.Parameter.VAR_POSITIONAL for name in signature.parameters):
        raise ActionError("'haystack_simple_action' can only handle functions without *args. Use a list instead.")
    if any(signature.parameters[name].kind == inspect.Parameter.VAR_KEYWORD for name in signature.parameters):
        raise ActionError(
            "'haystack_simple_action' can only handle functions without **kwargs. Use a dictionary instead."
        )

    # Filter out what the action expects
    filtered_data = {key: value for key, value in data.items() if key in signature.parameters}
    filtered_params = {key: value for key, value in parameters.get(name, {}).items()}

    action_kwargs = {**filtered_data, **filtered_params}
    logger.debug("%s is about to receive the following kwargs: %s", callable, action_kwargs)
    return action_kwargs
