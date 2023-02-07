from typing import Dict, Any, List
from functools import wraps
import logging
import inspect

from haystack.actions._utils import (
    DEFAULT_EDGE_NAME,
    ActionError,
    ActionValidationError,
    relevant_arguments,
)


logger = logging.getLogger(__name__)


def haystack_action(callable):
    """
    Bare minimum setup for Haystack actions. Any callable decorated with @haystack_action
    can be picked up by `find_actions` and be used in a Pipeline, serialized, deserialized, etc.

    Pipelines expects the following signature:

        `def my_action(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str], stores: Dict[str, Any])`

    If the callable is a class, the method used is `run`, which is expected to have the same signature
    plus a `self` argument at the start.

    Inputs have the following shape:
    ```
        data = {
            "documents": [Doc(), Doc()...],
            "files": [path.txt, path2.txt, ...],
            ...
        }

        parameters = {
            "node_1": {
                "param_1": 1,
                "param_2": 2,
            }
            "node_2": {
                "param_1": 1,
                "param_2": 3,
            }
            ...
        }

        outgoing_edges = [
            "edge_1",
            "another_edge",
            ...
        ]

        stores = {
            "my happy documents": <MemoryDocumentStore instance>,
            "testest-labels": <ESLabelStore instance>,
            "files": <S3FileStore instance>,
            ...
        }
    ```

    Pipelines expects the following output:

        `{output_edge: (relevant_data, relevant_parameters) for output_edge in output_edges}`

    Actions can therefore remove data from the pipeline and add/remove/alter parameter for EVERY following
    node by properly tweaking this output. Failing to produce output on one edge means that whatever node
    connects to it will receive no data and no parameters (and likely crash).

    Note that classes should also provide a `validate()` method if they want the class to be validated
    when warmed up.
    """
    logger.debug("Registering %s as a Haystack action", callable)

    # __haystack_action__ is used to tell Haystack Actions from regular functions.
    # Used by `find_actions`.
    # Set to the desired action name: normally the function/class name,
    # but could be customized.
    callable.__haystack_action__ = callable.__name__

    # Map run() to __call__() in classes
    # This way, run() can be used outside of pipelines unchanged
    # (Especially relevant for `haystack_simple_action`, used here too for consistency)
    if inspect.isclass(callable):
        if not hasattr(callable, "run"):
            raise ActionError(
                "Haystack class actions must have a run() method. See the docs for more information."
            )
        callable.__call__ = callable.run

        # Check for validate()
        if not hasattr(callable, "validate"):
            raise ActionError(
                "Haystack class actions must have a validate() method. See the docs for more information."
            )

    return callable


# TODO split the three wrappers to they can be reused separately in @haystack_node classes
def haystack_simple_action(callable):
    """
    Simplified API for Haystack actions. Any callable decorated with @haystack_simple_action
    can be picked up by `find_actions` and be used in a Pipeline, serialized, deserialized, etc.

    If the callable is a function, it can have any signature: the parameters are searched by name in
    the `data` and `parameter` dictionaries flowing down along the pipeline.

    if the callable is a class, the `run()` method is the one that the pipeline will invoke. It can have
    any signature just as in the case of the function. Note that classes should also provide a `validate()`
    method if they want the class to be validated when warmed up. However @haystack_simple_action already
    provides a basic validation method that checks for the presence of all and only the required init
    parameters in the `init_parameters` dictionary.

    Classes can have state and additional methods. However, consider keeping your actions lean to prevent
    issues with state serialization.

    Actions created with this decorator can output on one edge only. For nodes that can output on
    multiple edges, see @haystack_node.

    Actions created with this decorator can't access the stores dictionary. For nodes that can access them,
    see @haystack_node.
    """
    logger.debug("Registering %s as a Haystack simple action", callable)

    if inspect.isclass(callable):
        # class: we need to wrap run, assign the wrapped version to __call__, and pass self
        if not hasattr(callable, "run"):
            raise ActionError(
                "Haystack class actions must have a run() method. See the docs for more information."
            )
        run_method = callable.run

        @wraps(run_method)
        def run_wrapper(
            self,
            name: str,
            data: Dict[str, Any],
            parameters: Dict[str, Any],
            outgoing_edges: List[str],
            stores: Dict[str, Any],
        ):
            if outgoing_edges and any(
                edge != DEFAULT_EDGE_NAME for edge in outgoing_edges
            ):
                raise ActionError(
                    "'haystack_simple_action' can only output to one edge"
                )
            output = (
                run_method(
                    self, **relevant_arguments(run_method, name, data, parameters)
                )
                or {}
            )
            return {DEFAULT_EDGE_NAME: ({**data, **output}, parameters)}

        # Default 'validate()' that just checks that all mandatory args are there and no unknown args are given
        validate_method = (
            callable.validate
            if hasattr(callable, "validate")
            else lambda cls, init_parameters: None
        )

        @wraps(validate_method)
        def validate_wrapper(init_parameters: Dict[str, Any]):
            signature = inspect.signature(callable.__init__)

            # Check that all parameters given are in the signature
            # TODO check types too!
            for param_name in init_parameters.keys():
                if param_name not in signature.parameters.keys():
                    raise ActionValidationError(
                        f"{callable.__name__} does not expect a parameter called {param_name} in its init method."
                    )

            # Make sure all mandatory arguments are present
            for name in signature.parameters:
                if (
                    name != "self"
                    and signature.parameters[name].default == inspect.Parameter.empty
                    and name not in init_parameters
                ):
                    raise ActionValidationError(
                        f"{callable.__name__} requires a parameter called {name} in its init method."
                    )

            validate_method(init_parameters=init_parameters)

        # We need to also wrap __init__ to collect the init parameters by default
        init_method = callable.__init__

        @wraps(init_method)
        def init_wrapper(self, *args, **kwargs):
            if args:
                raise ActionError(
                    "'haystack_simple_action' does not support unnamed init parameters. "
                    "Pass all parameters as `MyAction(param_name=param_value)` instead of just `MyAction(param_value)`. "
                )
            self.init_parameters = kwargs
            init_method(self, **kwargs)

        callable.__call__ = run_wrapper
        callable.__init__ = init_wrapper
        callable.validate = validate_wrapper
        callable.__haystack_action__ = callable.__name__
        return callable

    # function: regular decorator pattern
    @wraps(callable)
    def wrapper(
        name: str,
        data: Dict[str, Any],
        parameters: Dict[str, Any],
        outgoing_edges: List[str],
        stores: Dict[str, Any],
    ):
        if outgoing_edges and any(edge != DEFAULT_EDGE_NAME for edge in outgoing_edges):
            raise ActionError("'haystack_simple_action' can only output to one edge")
        output = callable(**relevant_arguments(callable, name, data, parameters)) or {}
        return {DEFAULT_EDGE_NAME: ({**data, **output}, parameters)}

    wrapper.__haystack_action__ = callable.__name__
    return wrapper
