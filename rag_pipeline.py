import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import adalflow as adal
from dataclasses import dataclass

from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.core.types import Document

from adalflow.tracing import trace_generator_call, trace_generator_states
from process_ebooks import process_epub




@dataclass
class NarrativeTropeAnswer:
    answer: str
    citations: List[str]


###############################################################################
# Generator for Clarifying Questions
###############################################################################

class ClarifyingQuestionGenerator(adal.Generator):
    """
    Uses an LLM to generate a clarifying question when evidence is insufficient.
    The model is prompted to either propose a follow-up question or respond with "No further questions".
    """

    def __init__(self, model_client=None, model_kwargs=None):
        clarifying_template = r"""
Query: {{query}}
Evidence:
{{retrieved_context}}

If the above evidence is insufficient, propose a clarifying question to further investigate the narrative trope.
Otherwise, respond with "No further questions".
"""
        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=clarifying_template,
            prompt_kwargs={},  # No trainable parameters defined here.
        )


###############################################################################
# Generator for the Final Answer with Citations
###############################################################################

class FinalAnswerGenerator(adal.Generator):
    """
    Uses an LLM to produce the final answer along with citations.
    The answer is expected as a JSON object with "answer" and "citations" keys.

    A trainable prompt parameter "task_desc" is defined here so that the LLM evaluator
    can optimize the prompt during training.
    """

    def __init__(self, model_client=None, model_kwargs=None):
        answer_template = r"""
{{task_desc}}
Query: {{query}}

Combined Evidence:
{{context}}

Based on the above evidence, determine whether the narrative trope described in the query is present in the book.
Provide your answer as a JSON object with the following keys:
- "answer": A concise answer ("Yes" or "No").
- "confidence": A confidence score between 0 and 100.
- "explanation": A short explanation for your answer.
- "citations": A list of citations referencing the relevant book chunks.
"""
        prompt_params = {
            "task_desc": adal.Parameter(
                data="Summarize and answer whether the narrative trope exists in the story.",
                role_desc="Task description for final answer generation",
                param_type=adal.ParameterType.PROMPT,
                requires_opt=True,
            )
        }
        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=answer_template,
            prompt_kwargs=prompt_params,
        )


@trace_generator_call()
@trace_generator_states()
class QueryFocusedSummarizer(adal.Generator):
    """
    Iteratively summarizes the entire book (in batches of chunks) with a focus on details
    relevant to the query narrative trope. The summarization is performed in two stages:
    first summarizing batches of chunks, then combining these intermediate summaries into a final summary.
    """

    def __init__(self, model_client=None, model_kwargs=None, chunk_batch_size: int = 5):
        summary_template = r"""
You are provided with a group of book chunks and a query that specifies a narrative trope.
Summarize the key details from the following text that are relevant to answering whether the narrative trope is present.
Query: {{query}}
Text:
{{chunk_text}}

Provide a concise summary focusing on narrative elements and details related to the query.
Final summary:
"""
        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=summary_template,
            prompt_kwargs={},
        )
        self.chunk_batch_size = chunk_batch_size

    def call(
        self, prompt_kwargs: Dict, id: Optional[str] = None
    ) -> adal.GeneratorOutput:
        # Expected prompt_kwargs: {"query": query, "book_chunks": List[Dict[str, str]]}
        query = prompt_kwargs.get("query", "")
        book_chunks = prompt_kwargs.get("book_chunks", [])

        batch_summaries = []
 
        for i in range(0, len(book_chunks), self.chunk_batch_size):
            batch = book_chunks[i : i + self.chunk_batch_size]
            batch_text = "\n\n".join(chunk["text"] for chunk in batch)
            new_prompt_kwargs = {"query": query, "chunk_text": batch_text}
            batch_summary_output = super().call(prompt_kwargs=new_prompt_kwargs, id=id)
            batch_summary = (
                batch_summary_output.data if batch_summary_output.data else ""
            )
            batch_summaries.append(batch_summary)

        # Combine batch summaries and produce a final focused summary.
        combined_summaries = "\n\n".join(batch_summaries)
        final_prompt_kwargs = {"query": query, "chunk_text": combined_summaries}
        final_summary_output = super().call(prompt_kwargs=final_prompt_kwargs, id=id)
        final_summary = final_summary_output.data if final_summary_output.data else ""
        return adal.GeneratorOutput(data=final_summary)


###############################################################################
# The Iterative RAG Pipeline for Narrative Trope Evaluation with FAISS
###############################################################################

@trace_generator_call()
@trace_generator_states()
class NarrativeTropeRAG(adal.GradComponent):
    """
    Implements an iterative RAG pipeline that:
      1. Uses an embedding-based FAISS retriever to retrieve relevant book chunks.
      2. Optionally refines the query via clarifying questions.
      3. Summarizes the entire book chunk-by-chunk with focus on the query.
      4. Combines all evidence and generates a final answer with citations.
    """

    def __init__(
        self,
        desc: str,
        model_client=None,
        model_kwargs=None,
        passages_per_hop: int = 3,
        max_iterations: int = 3,
    ):
        super().__init__(desc=desc)
        self.passages_per_hop = passages_per_hop
        self.max_iterations = max_iterations

       
        self.embedder = adal.Embedder(
            model_client=adal.OllamaClient(host="http://localhost:11434"),
            model_kwargs={"model": "bge-m3:latest"},
        )

       
        self.faiss_retriever = FAISSRetriever(
            top_k=passages_per_hop,
            embedder=self.embedder,
            dimensions=1024,
            documents=[],  # to be built from the input book chunks
            document_map_func=lambda doc: doc.vector,
        )

     
        self.question_generator = ClarifyingQuestionGenerator(
            model_client=model_client, model_kwargs=model_kwargs
        )
        self.summarizer = QueryFocusedSummarizer(
            model_client=model_client, model_kwargs=model_kwargs, chunk_batch_size=5
        )
        self.answer_generator = FinalAnswerGenerator(
            model_client=model_client, model_kwargs=model_kwargs
        )

    def _prepare_documents(self, book_chunks: List[Dict[str, str]]) -> List[Document]:
        """
        Converts book chunks (dict with 'text' and 'citation') into a list of Document objects.
        """
        docs = []
        for chunk in book_chunks:
            docs.append(
                Document(
                    id=chunk.get("citation", ""),
                    text=chunk["text"],
                    meta_data=chunk,  # store entire chunk as metadata
                )
            )
        return docs

    def _retriever_context(
        self, retriever_outputs: List[Any], docs: List[Document]
    ) -> str:
        """
        Given FAISS retriever outputs and the original document list,
        returns a combined context string (with citations) from the retrieved documents.
        """
        if (
            retriever_outputs
            and retriever_outputs[0]
            and retriever_outputs[0].doc_indices
        ):
            retrieved_docs = [docs[idx] for idx in retriever_outputs[0].doc_indices]
            documents = [
                f"{doc.text} [Citation: {doc.meta_data.get('citation', doc.id)}]"
                for doc in retrieved_docs
            ]
            return "\n\n".join(documents)
        return ""

    def call(
        self, query: str, book_chunks: List[Dict[str, str]], id: Optional[str] = None
    ) -> adal.GeneratorOutput:
        if self.training:
            raise ValueError(
                "This component is not supposed to be called in training mode"
            )

        # Convert book chunks to Document objects and build the FAISS index.
        docs = self._prepare_documents(book_chunks)
        self.faiss_retriever.build_index_from_documents(
            docs,
            document_map_func=lambda doc: self.embedder(doc.text).data[0].embedding,
        )

        memory = [] 
        current_query = query
        iteration = 0

        while iteration < self.max_iterations:
            retriever_outputs = self.faiss_retriever(current_query)
            retrieved_context = self._retriever_context(retriever_outputs, docs)

            if retrieved_context:
                memory.append(retrieved_context)

            # Generate a clarifying question to refine the search.
            clarifying_prompt_kwargs = {
                "query": current_query,
                "retrieved_context": retrieved_context,
            }
            clarifying_output = self.question_generator.call(
                prompt_kwargs=clarifying_prompt_kwargs, id=id
            )
            clarifying_question = (
                clarifying_output.data.strip()
                if clarifying_output and clarifying_output.data
                else ""
            )
            if clarifying_question.lower() == "no further questions":
                break

            current_query = clarifying_question
            iteration += 1

    
        summary_output = self.summarizer.call(
            prompt_kwargs={"query": query, "book_chunks": book_chunks}, id=id
        )
        summary_text = summary_output.data

      
        combined_context = "\n\n".join(memory + [summary_text])
        final_prompt_kwargs = {
            "query": query,
            "context": combined_context,
        }
        final_output = self.answer_generator.call(
            prompt_kwargs=final_prompt_kwargs, id=id
        )
        return final_output

    def forward(
        self, query: str, book_chunks: List[Dict[str, str]], id: Optional[str] = None
    ) -> adal.Parameter:
        raise NotImplementedError("Training mode not implemented for NarrativeTropeRAG")


###############################################################################
# AdalComponent for the Narrative Trope Task (for integration with Trainer)
###############################################################################
class NarrativeTropeAdal(adal.AdalComponent):
    """
    Wraps the NarrativeTropeRAG pipeline for training/evaluation.
    Defines how to prepare input samples and how to parse output for evaluation.
    """

    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        passages_per_hop: int = 3,
    ):
        task = NarrativeTropeRAG(
            desc="Narrative Trope RAG Pipeline",
            model_client=model_client,
            model_kwargs=model_kwargs,
            passages_per_hop=passages_per_hop,
        )
        eval_fn = (
            lambda y, y_gt: 1.0 if y.strip().lower() == y_gt.strip().lower() else 0.0
        )
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="Exact match between generated answer and ground truth answer",
        )
        super().__init__(task=task, eval_fn=eval_fn, loss_fn=loss_fn)

    def prepare_task(self, sample: Dict[str, Any]) -> Tuple[Callable[..., Any], Dict]:
        # Sample is expected to contain "query", "book_chunks", "answer", and optionally "id".
        if self.task.training:
            return self.task.forward, {
                "query": sample["query"],
                "book_chunks": sample["book_chunks"],
                "id": sample.get("id", None),
            }
        else:
            return self.task.call, {
                "query": sample["query"],
                "book_chunks": sample["book_chunks"],
                "id": sample.get("id", None),
            }

    def prepare_eval(
        self, sample: Dict[str, Any], y_pred: adal.GeneratorOutput
    ) -> float:
        y_label = ""
        if y_pred and y_pred.data:
            try:
                parsed = json.loads(y_pred.data)
                y_label = parsed.get("answer", "")
            except json.JSONDecodeError:
                y_label = y_pred.data
        return self.eval_fn(y=y_label, y_gt=sample["answer"])

    def prepare_loss(self, sample: Dict[str, Any], pred: adal.Parameter):
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample["answer"],
            eval_input=sample["answer"],
            requires_opt=False,
        )
        pred.eval_input = (
            pred.full_response.data.get("answer", "")
            if pred.full_response and hasattr(pred.full_response, "data")
            else ""
        )
        return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt}}


###############################################################################
# Example Usage
###############################################################################


def test_narrative_trope_pipeline(book_chunks=None, user_query=None):
    if book_chunks is None:
        book_chunks = [
            {
                "text": "In the dark corridors of the ancient castle, a mysterious figure roamed, embodying the haunted hero trope.",
                "citation": "Chapter 3",
            },
            {
                "text": "The protagonist often faced overwhelming odds, a common trope in epic sagas.",
                "citation": "Chapter 5",
            },
            {
                "text": "A subtle reference to a tragic love story was hidden in the dialogue.",
                "citation": "Chapter 7",
            },
            
        ]
    if user_query is None:
        
        user_query = "Does the book contain the haunted hero trope?"

   
    model_client = adal.OllamaClient(host="http://localhost:11434") .
    model_kwargs = {
        "model": "qwen2.5:14b-instruct-q8_0"
    }  

  
    adal_component = NarrativeTropeAdal(
        model_client=model_client, model_kwargs=model_kwargs, passages_per_hop=2
    )

    
    sample = {
        "query": user_query,
        "book_chunks": book_chunks,
        "answer": "Yes",  
        "id": "sample_001",
    }

  
    task_fn, task_kwargs = adal_component.prepare_task(sample)
    result = task_fn(**task_kwargs)
    print("Final output:", result.data)


def test_on_book(book_id):
    book_chunks = process_epub(f"{book_id}.epub", book_id, max_chunk_length=100)
    user_query = "Does the book contain this trope example: Blackmail: A guest caught by a hotel employee doing something he shouldn't leaves a bigger tip, but the only person to interpret it as a blackmailing gesture is the murderer, who isn't a local."
    test_narrative_trope_pipeline(book_chunks=book_chunks, user_query=user_query)


if __name__ == "__main__":
    test_on_book("lit20")
