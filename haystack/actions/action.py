from typing import Dict, Any, List
from functools import wraps
import logging
import inspect

from haystack.actions._utils import DEFAULT_EDGE_NAME, ActionError, relevant_arguments


logger = logging.getLogger(__name__)


def haystack_action(callable):
    logger.debug("Registering %s as a Haystack action", callable)
    callable.__haystack_action__ = callable.__name__
    return callable



# TODO Implement default validation for the input parameters (check that all mandatory keys exist)
# Remember that validate() it's a CLASS METHOD
def haystack_simple_action(callable):
    logger.debug("Registering %s as a Haystack simple action", callable)

    if inspect.isclass(callable):
        # class: we need to wrap __call__ and pass self
        call_method = callable.__call__

        @wraps(call_method)
        def call_wrapper(self, name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
            if outgoing_edges and outgoing_edges != [DEFAULT_EDGE_NAME]:
                raise ActionError("'haystack_simple_action' can only output to one edge")
            output = call_method(self, **relevant_arguments(call_method, name, data, parameters)) or {}
            return {DEFAULT_EDGE_NAME: ({**data, **output}, parameters)}

        # We need to also wrap __init__ to collect the init parameters by default
        init_method = callable.__init__
        @wraps(init_method)
        def init_wrapper(self, *args, **kwargs):
            if args:
                raise ActionError(
                    "'haystack_simple_action' does not support unnamed init parameters. "
                    "Pass all parameters as `MyAction(param_name=param_value)` instead of just `MyAction(param_value)`. ")
            self.init_parameters = kwargs
            init_method(self, **kwargs)
            
        callable.__call__ = call_wrapper
        callable.__init__ = init_wrapper
        callable.__haystack_action__ = callable.__name__
        return callable

    # function: regular decorator pattern
    @wraps(callable)
    def wrapper(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
        if outgoing_edges and outgoing_edges != [DEFAULT_EDGE_NAME]:
            raise ActionError("'haystack_simple_action' can only output to one edge")
        output = callable(**relevant_arguments(callable, name, data, parameters)) or {}
        return {DEFAULT_EDGE_NAME: ({**data, **output}, parameters)}

    wrapper.__haystack_action__ = callable.__name__
    return wrapper
