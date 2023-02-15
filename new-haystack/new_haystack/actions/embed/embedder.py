from typing import Dict, Set, Any, Optional, List

import logging

from tqdm import tqdm

from new_haystack.actions import haystack_node, ActionError
from new_haystack.data import Data


logger = logging.getLogger(__name__)


try:
    import torch
except ImportError as e:
    logger.debug(
        "torch not found:Embedder won't be able to use local models."
    )

try:
    import sentence_transformers
except ImportError as e:
    logger.debug(
        "sentence_transformers not found: Embedder won't be able to use local models."
    )


class EmbedderError(ActionError):
    pass


@haystack_node
class Embedder:
    """
    Adds embeddings to a list of Data objects.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        embed_meta_fields: List[str] = ["name"],
        model_params: Optional[Dict[str, Dict[str, Any]]] = None,
        progress_bar: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **(model_params or {}))
        self.progress_bar = progress_bar
        self.embed_meta_fields = embed_meta_fields

    def run(
        self,
        name: str,
        data: Dict[str, Any],
        parameters: Dict[str, Any],
        outgoing_edges: Set[str],
        stores: Dict[str, Any],
    ):
        input_variable_name = parameters.get(name, {}).get("input", "data")
        output_variable_name = parameters.get(name, {}).get("output", "data")

        # FIXME Fail or log?
        if not input_variable_name in data.keys():
            logger.error(
                "No data to compute embeddings for! '%s' is not present in the pipeline context. "
                "'%s' won't add any embedding and won't output anything on '%s'.",
                input_variable_name,
                name,
                output_variable_name
            )

        # It's polite to pop your input
        input_items = data.pop(input_variable_name)

        if isinstance(input_items, Data):
            input_items = [input_items]
        elif not (isinstance(input_items, list) and input_items and isinstance(input_items[0], Data)):
            raise EmbedderError("Pass a list of Data objects to the Embedder.")
            
        output_items = []
        for input_item in tqdm(
            input_items, unit=" units", desc="Creating embeddings..."
        ):
            to_embed = " ".join([input_item.content] + [input_item.meta.get(key, "") for key in self.embed_meta_fields])
            embedding = self.model.encode(to_embed)

            data_type = input_item.__class__
            dict_repr = input_item.to_dict()
            dict_repr["embedding"] = embedding
            dict_repr["meta"] = {**dict_repr.get("meta", {}), "embedding_model": self.model_name}

            output_items.append(data_type.from_dict(dict_repr))

        data[output_variable_name] = output_items

        return {edge: (data, parameters) for edge in outgoing_edges}

    @staticmethod
    def validate(init_parameters: Dict[str, Any]) -> None:
        # We could check if the model name exists and if it's the proper model type.
        pass