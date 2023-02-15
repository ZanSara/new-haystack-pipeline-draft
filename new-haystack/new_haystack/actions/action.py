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
            "input_edge1:": {
                "documents": [Doc(), Doc()...],
                "files": [path.txt, path2.txt, ...],
                "whatever": ...
                ...
            },
            "input_edge2": { ... }
        }

        parameters = {
            "input_edge1": {
                "node_1": {
                    "param_1": 1,
                    "param_2": 2,
                }
                "node_2": {
                    "param_1": 1,
                    "param_2": 3,
                }
            }
            "input_edge2": { ... }
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
        {
            "output_edge_1": (relevant_data, all_parameters),
            "output_edge_2": ...
        }
    ```

    Nodes can remove data from the pipeline and add/remove/alter parameter for EVERY following
    node by properly tweaking this output. Failing to produce output on one edge means that whatever node
    connects to it will receive no data and no parameters.

    Sending no data and no parameters to a node means that it will not run: see Pipeline for details.

    Nodes must also provide a `validate()` method. Validation expects nodes to have the following instance attributes:

    ```
    self.expects_inputs = {
        "my_input_edge1": {"documents"},
        "my_input_edge2": {"query"},
        ...
    }
    self.produces_outputs = {
        "my_output_edge1": {"answers"},
        ...
    }
    ```
    ##############
    # UNNECESSARY?
    #
    # The `validate()` instance method with the following signature:
    
    # ```
    # def validate(
    #     self,
    #     receiving_input: Dict[str, Set[str]],
    #     expected_output: Dict[str, Set[str]]
    # ):
    # ```

    # This method should check whether the node is properly connected to nodes that produce the 
    # output it expects, and that it's connected to enough downstream nodes that expect what 
    # the node is outputting.
    
    # Remember to consider corner cases such as your node being at the start of the pipeline
    # (no input edges) or at the end of the pipeline (no output edges).
    """
    logger.debug("Registering %s as a Haystack action", callable)

    # __haystack_action__ is used to tell Haystack Actions from regular functions.
    # Used by `find_actions`.
    # Set to the desired action name: normally the function/class name,
    # but could be customized.
    callable.__haystack_action__ = callable.__name__

    # Check for run()
    if not hasattr(callable, "run"):
        raise ActionError(
            "Haystack actions must have a 'run()' method. See the docs for more information."
        )

    # Check for validate()
    if not hasattr(callable, "validate"):
        raise ActionError(
            "Haystack nodes must have a 'validate()' method. See the docs for more information."
        )

    return callable
