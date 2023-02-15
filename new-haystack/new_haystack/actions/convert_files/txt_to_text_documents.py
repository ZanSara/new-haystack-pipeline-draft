from typing import Dict, List, Any

import logging

from new_haystack.actions import haystack_node, ActionError
from new_haystack.data import TextDocument


logger = logging.getLogger(__name__)


@haystack_node
def convert_txt_to_text_documents(
    name: str,
    data: Dict[str, Any],
    parameters: Dict[str, Any],
    outgoing_edges: List[str],
    stores: Dict[str, Any],
):
    input_variable_name = parameters.get(name, {}).get("input", "txt_files")
    output_variable_name = parameters.get(name, {}).get("output", "documents")
    file_loading_parameters = parameters.get(name, {}).get("file opening options", {})

    # FIXME Fail or log?
    if not input_variable_name in data.keys():
        logger.error(
            "No files to convert! '%s' is not present in the pipeline context. "
            "'%s' won't convert any file and output nothing under %s.",
            input_variable_name,
            name,
            output_variable_name
        )

    # We decide it's polite for a node to pop its input, so that less data
    # keeps flowing down the pipeline.
    files_to_convert = data.pop(input_variable_name)
    documents = []
    for file_to_convert in files_to_convert:
        try:
            with open(file_to_convert, 'r', **file_loading_parameters) as file_handle:
                content = file_handle.read()
                document = TextDocument(content=content, meta={"name": file_to_convert})
                documents.append(document)
        except Exception as e:
            logger.exception("Failed to convert '%s'. Dropping it.", files_to_convert)

    data[output_variable_name] = documents

    # TBD: output on all edges the same thing, or enforce everyone to connect
    # to the same "default" edge? For now using strategy one.
    return {edge: (data, parameters) for edge in outgoing_edges}
