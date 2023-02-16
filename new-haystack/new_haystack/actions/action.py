import logging

from new_haystack.actions._utils import ActionError


logger = logging.getLogger(__name__)


def haystack_node(callable):
    """
    Bare minimum setup for Haystack nodes. Any class decorated with @haystack_node
    can be picked up by `discover_nodes` and be used in a Pipeline, serialized, deserialized, etc.

    Pipelines expects the following signature:

    ```
        def run(
            self,
            name: str, 
            data: Dict[str, Any], 
            parameters: Dict[str, Any],
            stores: Dict[str, Any]
        ):
    ```
    
    Inputs have the following shape:
    
    ```
        data = {
            "documents": [Doc(), Doc()...],
            "files": [path.txt, path2.txt, ...],
            "whatever": ...
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
            },
            ...
        }

        stores = {
            "my happy documents": <MemoryDocumentStore instance>,
            "test-labels": <ESLabelStore instance>,
            "files-store": <S3FileStore instance>,
            ...
        }
    ```

    Pipelines expects the following output:

    ```
        ({output_edge: value, ...}, all_parameters)
    ```

    Nodes can add/remove/alter parameter for EVERY following node.
    Failing to produce output for one edge means that whatever node connects to it will receive 
    no data and no parameters, and sending no data and no parameters to a node means that it will not run: 
    see Pipeline for details.

    Pipeline.connect() performs some basic validation. It expects nodes to have the following instance attributes:

    ```
    self.expected_inputs = ["documents", "query", ...]
    self.expected_outputs = ["answers", ...]
    ```

    """
    logger.debug("Registering %s as a Haystack action", callable)

    # __haystack_action__ is used to tell Haystack Actions from regular functions.
    # Used by `find_actions`. Set to the desired action name: normally the function/class 
    # name, but could be customized.
    callable.__haystack_action__ = callable.__name__

    # Check for run()
    if not hasattr(callable, "run"):
        raise ActionError(
            "Haystack actions must have a 'run()' method. See the docs for more information."
        )

    return callable
