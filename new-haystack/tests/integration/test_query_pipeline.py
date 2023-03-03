from typing import Dict, Any, List, Tuple

import json
from pathlib import Path

from new_haystack.pipeline import Pipeline
from new_haystack.stores import MemoryDocumentStore
from new_haystack.data import TextQuery, TextDocument, TextAnswer, Span
from new_haystack.nodes import haystack_node

import logging

logging.basicConfig(level=logging.DEBUG)


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
    

@haystack_node
class ReaderByTransformers:
    """
    Simple dummy Transformers Reader.
    Supports batch processing.
    """
    def __init__(self, 
        model_name_or_path: str,
        default_top_k: int = 5,
        default_no_answer: bool = True,
        default_max_seq_len: int = 256,
        default_doc_stride: int = 128,
        default_batch_size: int = 16,
        default_context_window_size: int = 70,
        input_name: str = "documents_by_query",
        output_name: str = "answers_by_query", 
    ):
        self.model_name_or_path = model_name_or_path
        self.default_top_k = default_top_k
        self.default_no_answer = default_no_answer
        self.default_max_seq_len = default_max_seq_len
        self.default_doc_stride = default_doc_stride
        self.default_batch_size = default_batch_size
        self.default_context_window_size = default_context_window_size
        self.model = None

        self.init_parameters = {
            "input_name": input_name, 
            "output_name": output_name, 
            "model_name_or_path": model_name_or_path, 
            "default_top_k": default_top_k,
            "default_no_answer": default_no_answer,
            "default_max_seq_len": default_max_seq_len,
            "default_doc_stride": default_doc_stride,
            "default_batch_size": default_batch_size,
            "default_context_window_size": default_context_window_size,
        }
        self.expected_inputs = [input_name]
        self.expected_outputs = [output_name]

    def warm_up(self):
        try:
            from transformers import pipeline
        except Exception as e:
            raise ImportError("Can't import 'transformers': this node won't work.") from e

        self.model = pipeline(
            "question-answering",
            model=self.model_name_or_path,
        )

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        my_parameters = parameters.get(name, {})
        top_k = my_parameters.pop("top_k", self.default_top_k)
        no_answer = my_parameters.pop("no_answer", self.default_no_answer)
        max_seq_len = my_parameters.pop("max_seq_len", self.default_max_seq_len)
        doc_stride = my_parameters.pop("doc_stride", self.default_doc_stride)
        batch_size = my_parameters.pop("batch_size", self.default_batch_size)
        context_window_size = my_parameters.pop("context_window_size", self.default_context_window_size)

        documents_for_queries = data[0][1]

        inputs = []
        for query, documents in documents_for_queries.items():
            inputs.extend([
                self.model.create_sample(question=query.content, context=doc.content)  # type: ignore
                for doc in documents
            ])

        # Inference
        predictions = self.model(   # type: ignore
            inputs,
            top_k=top_k,
            handle_impossible_answer=no_answer,
            max_seq_len=max_seq_len,
            doc_stride=doc_stride,
            batch_size=batch_size,
        )

        # Builds the TextAnswer object
        answers_for_queries = {query: [] for query in documents_for_queries.keys()}
        for query, documents in documents_for_queries.items():
            documents = list(documents) # FIXME consume here the iterator for now
            docs_len = len(documents)
            relevant_predictions = predictions[:docs_len]
            predictions = predictions[docs_len:]

            for document, prediction in zip(documents, relevant_predictions):
                if prediction.get("answer", None):
                    context_start = max(0, prediction["start"] - context_window_size)
                    context_end = min(len(document.content), prediction["end"] + context_window_size)
                    answers_for_queries[query].append(
                        TextAnswer(
                            content=prediction["answer"],
                            score=prediction["score"],
                            context=document.content[context_start:context_end],
                            offset_in_document=Span(start=prediction["start"], end=prediction["end"]),
                            offset_in_context=Span(start=prediction["start"] - context_start, end=prediction["end"] - context_start),
                            document_id=document.id,
                            meta=document.meta,
                        )
                    )
                elif no_answer:
                    answers_for_queries[query].append(
                        TextAnswer(
                            content="",
                            score=prediction["score"],
                            meta=document.meta,
                        )
                    )
            answers_for_queries[query] = sorted(answers_for_queries[query], reverse=True)[:top_k]
        return {self.expected_outputs[0]: answers_for_queries}


def test_pipeline(tmp_path):
    document_store = MemoryDocumentStore()
    document_store.write_documents([
        TextDocument(content="My name is Anna and I live in Paris."),
        TextDocument(content="My name is Serena and I live in Rome."),
        TextDocument(content="My name is Julia and I live in Berlin."),
    ])

    pipeline = Pipeline()
    pipeline.connect_store("my_documents", document_store)
    pipeline.add_node("retriever", RetrieveByBM25(default_store="my_documents"))
    pipeline.add_node("reader", ReaderByTransformers(model_name_or_path="distilbert-base-uncased-distilled-squad"))

    pipeline.connect(["retriever", "reader"])
    pipeline.draw(tmp_path / "query_pipeline.png")

    results = pipeline.run({"query": TextQuery(content="Who lives in Berlin?")})

    results["answers_by_query"] = {str(key): value for key, value in results["answers_by_query"].items()}
    print(json.dumps(results, indent=4, default=repr))


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
