import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

import adalflow as adal
from dataclasses import dataclass, field
from adalflow.core.types import DataClass

from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.core.types import Document

from adalflow.tracing import trace_generator_call, trace_generator_states
from m3_embedder import M3Embedder
from embedder import FastAPIClientEmbedder
from process_ebooks import load_jsonl
from adalflow import get_logger
from rate_limiter import rate_limited_call

import dotenv

dotenv.load_dotenv()
logger = get_logger(__name__, level="DEBUG")


confidence_rubric = """###Confidence score rubrics:
[Does the evidence trace lead to a conclusion that is relevant to the query asked by the user?]
Score 1: Not relevant to the query
Score 2: Somewhat relevant to the query
Score 3: Relevant to the query
Score 4: Highly relevant to the query
Score 5: A perfect match to the query
"""



@dataclass
class TropeQuery(DataClass):
    id: str
    book_id: str
    trope_name: str
    trope_definition: str
    trope_example: str
    query: str
    answer: bool


@dataclass
class CitationStore:
    citations: Dict[str, str] = field(
        default_factory=dict,
        metadata={
            "description": "A dictionary of citations where they are mapped by their citation id to their corresponding text."
        },
    )

    def add_by_doc_indices(
        self, docs: List[Document], doc_indices: List[int]
    ) -> List[str]:
        return [self.add_citation(doc.id, doc.text) for doc in docs[doc_indices]]

    def get_citation(self, citation_id: str) -> str:
        return self.citations[citation_id]

    def get_citations(self, citation_ids: List[str]) -> List[str]:
        return [
            f"<Citation id: {citation_id}>\nText: {self.citations[citation_id]}</Citation>"
            for citation_id in citation_ids
        ]

    def add_citation(self, citation_id: str, citation: str):
        self.citations[citation_id] = citation

    def add_citations(self, citation_ids: list[str], citations: list[str]):
        self.citations.update(zip(citation_ids, citations))


@dataclass
class Question(DataClass):
    id: str = field(
        default_factory=str,
        metadata={"description": "The id of the question."},
    )

    text: str = field(
        default_factory=str,
        metadata={
            "description": "The original input user query that the evidence tree is answering."
        },
    )

    reasoning: str = field(
        default_factory=str,
        metadata={
            "description": "The step-by-step reasoning chain behind why the question should be asked and how it contributes to the overall goal of the question. Each step of the reasoning chain should be short and include most influential keywords."
        },
    )

    retrieved_context: List[Document] = field(
        default_factory=list,
        metadata={
            "description": "The context that supports the answer that was retrieved by the retriever (semantic vector search)."
        },
    )

    dependencies: List[str] = field(
        default_factory=list,
        metadata={
            "description": "The dependencies of the question. These are ids of questions that are used to answer the current question."
        },
    )
    __output_fields__ = ["text", "reasoning", "dependencies"]


@dataclass
class QuestionGeneratorAnswer(DataClass):
    questions: List[Question] = field(
        default_factory=list,
        metadata={
            "description": "The questions that are necessary to find key paragraphs by semenatic search that can be used to fully and accurately answer the original question."
        },
    )
    __input_fields__ = ["questions"]
    __output_fields__ = ["questions"]


@dataclass
class QueryRephraseResult(DataClass):
    rephrased_queries: List[str] = field(
        default_factory=list,
        metadata={
            "description": "A list of rephrased queries that are semantically similar to the original query but optimized for semantic search to find relevant passages about tropes in books."
        },
    )
    reasoning: str = field(
        default_factory=str,
        metadata={
            "description": "The reasoning behind the rephrased queries, explaining how they help find relevant passages."
        },
    )
    __input_fields__ = []
    __output_fields__ = ["rephrased_queries", "reasoning"]


@dataclass
class InvestigationPath(DataClass):
    id: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={"description": "Unique identifier for this investigation path."},
    )
    description: str = field(
        default_factory=str,
        metadata={
            "description": "Brief description of what this investigation path is exploring."
        },
    )
    reasoning: str = field(
        default_factory=str,
        metadata={
            "description": "Reasoning for why this path might lead to evidence of the trope."
        },
    )
    search_queries: List[str] = field(
        default_factory=list,
        metadata={
            "description": "List of questions that are used to find relevant passages about the trope."
        },
    )
    priority: int = field(
        default=1,
        metadata={
            "description": "Priority of this investigation path (1-5, with 1 being highest)."
        },
    )
    status: str = field(
        default="pending",
        metadata={
            "description": "Status of this investigation path: 'pending', 'in_progress', 'completed'."
        },
    )
    __input_fields__ = []
    __output_fields__ = [
        "id",
        "description",
        "reasoning",
        "search_queries",
        "priority",
        "status",
    ]


@dataclass
class InvestigationPlan(DataClass):
    paths: List[InvestigationPath] = field(
        default_factory=list,
        metadata={"description": "List of investigation paths to explore."},
    )
    reasoning: str = field(
        default_factory=str,
        metadata={"description": "Overall reasoning for the investigation plan."},
    )
    __input_fields__ = []
    __output_fields__ = ["paths", "reasoning"]


@dataclass
class PathInvestigationResult(DataClass):
    path_id: str = field(
        default_factory=str,
        metadata={"description": "ID of the investigation path that was explored."},
    )
    conclusion: str = field(
        default_factory=str,
        metadata={
            "description": "Conclusion about whether this path provides evidence for the trope: 'supports', 'contradicts', 'inconclusive'."
        },
    )
    confidence: int = field(
        default=0,
        metadata={
            "description": "Confidence in the conclusion (1-5, with 5 being highest)."
        },
    )
    reasoning: str = field(
        default_factory=str,
        metadata={
            "description": "Detailed reasoning for the conclusion based on the evidence."
        },
    )
    evidence: List[str] = field(
        default_factory=list,
        metadata={"description": "List of citation IDs that support the conclusion."},
    )

    follow_up_paths: list[InvestigationPath] = field(
        default_factory=None,
        metadata={
            "description": "Any new investigation paths suggested based on this investigation. This list should contain list of 'properties' in the output format."
        },
    )
    __input_fields__ = []
    __output_fields__ = [
        "path_id",
        "conclusion",
        "confidence",
        "reasoning",
        "evidence",
        "follow_up_paths",
    ]


@dataclass
class FinalTropeAnswer(DataClass):
    answer: str = field(
        default_factory=str,
        metadata={"description": "Final answer: 'Yes', 'No', or 'Inconclusive'."},
    )
    confidence: int = field(
        default=0,
        metadata={
            "description": "Confidence in the answer (1-5, with 5 being highest)."
        },
    )
    reasoning: str = field(
        default_factory=str,
        metadata={
            "description": "Detailed reasoning for the final answer, synthesizing all investigation paths."
        },
    )
    key_evidence: List[str] = field(
        default_factory=list,
        metadata={
            "description": "List of the most important citation IDs that support the final answer."
        },
    )
    __input_fields__ = []
    __output_fields__ = ["answer", "confidence", "reasoning", "key_evidence"]


###############################################################################
# Generator for Query Rephrasing
###############################################################################
@trace_generator_states()
@trace_generator_call(error_only=False)
class QueryRephraseGenerator(adal.Generator):
    def __init__(self, model_client=None, model_kwargs=None):
        rephrase_template = r"""
{{task_desc}}
<output_format>
{{output_format_str}}
</output_format>
<trope_query>
{{query}}
</trope_query>
"""
        parser = adal.DataClassParser(
            data_class=QueryRephraseResult,
            format_type="json",
            return_data_class=True,
        )
        prompt_params = {
            "output_format_str": parser.get_output_format_str(),
            "task_desc": adal.Parameter(
                data="""You are a literary search expert specialized in finding narrative tropes, plot elements, and character traits in books. Your task is to rephrase the user's query into multiple search queries that will be more effective for semantic search to find relevant passages.

INSTRUCTIONS:
1. ANALYZE the user's query to identify the specific narrative trope, plot element, or character trait they're looking for.
2. GENERATE 8-12 rephrased queries that will help a semantic search engine find relevant passages in a book.
3. ENSURE each query focuses on different aspects or manifestations of the trope.
4. PROVIDE a brief reasoning explaining your approach to rephrasing the queries.

EFFECTIVE QUERY REPHRASING STRATEGIES:

1. DECOMPOSE COMPLEX TROPES into their constituent elements:
   - Break down multi-part tropes into individual components
   - Create separate queries for different manifestations of the trope
   - Focus on specific character dynamics, plot points, or narrative patterns

2. USE LITERARY TERMINOLOGY AND COMMON EXPRESSIONS:
   - Include both formal literary terms and colloquial expressions
   - Use phrases that authors might use to describe the trope
   - Include metaphors or similes commonly associated with the trope

3. FOCUS ON OBSERVABLE ELEMENTS in text:
   - Character actions, dialogue, and reactions
   - Narrative descriptions and scene settings
   - Emotional states and character development

4. VARY LINGUISTIC PATTERNS:
   - Use different syntactic structures
   - Employ both direct and indirect descriptions
   - Include both questions and statements

5. INCORPORATE CONTEXTUAL ELEMENTS:
   - Consider historical or genre-specific manifestations
   - Include setting-specific variations
   - Address cultural or social dimensions

6. CONSIDER CHARACTER PERSPECTIVES:
   - How different characters might experience or perceive the trope
   - Dialogue or thoughts that might reveal the trope
   - Relationships between characters that embody the trope

Your rephrased queries should be specific enough to find relevant passages but general enough to capture different expressions of the trope. Focus on how the trope might be described in the text rather than abstract definitions.
""",
                param_type=adal.ParameterType.PROMPT,
                requires_opt=True,
            ),
        }

        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=rephrase_template,
            prompt_kwargs=prompt_params,
            output_processors=parser,
        )


###############################################################################
# Generator for Investigation Planning
###############################################################################
@trace_generator_states()
@trace_generator_call(error_only=False)
class InvestigationPlanGenerator(adal.Generator):
    def __init__(self, model_client=None, model_kwargs=None):
        plan_template = r"""
{{task_desc}}
<output_format>
{{output_format_str}}
</output_format>
<trope_query>
{{query}}
</trope_query>
<initial_findings>
{{initial_findings}}
</initial_findings>
"""
        parser = adal.DataClassParser(
            data_class=InvestigationPlan,
            format_type="json",
            return_data_class=True,
        )
        prompt_params = {
            "output_format_str": parser.get_output_format_str(),
            "task_desc": adal.Parameter(
                data="""You are a literary detective specialized in identifying narrative tropes in books. Your task is to create a comprehensive investigation plan to determine whether a specific trope is present in a book, based on initial findings from semantic search.

INSTRUCTIONS:
1. ANALYZE the trope query and initial findings to identify potential evidence of the trope.
2. DEVELOP 3-5 distinct investigation paths that could lead to conclusive evidence.
3. For each path, provide:
   - A clear description of what this path is investigating
   - Reasoning for why this path might lead to evidence of the trope
   - 2-4 specific search queries to explore this path, provide information from the found and potentially usefull paragraphs such as specific book scenes, events, characters
   - A priority level (1-5, with 1 being highest priority)

EFFECTIVE INVESTIGATION PLANNING STRATEGIES:

1. IDENTIFY KEY CHARACTERS AND RELATIONSHIPS:
   - Characters who might embody or experience the trope
   - Relationships that might manifest the trope
   - Character development arcs that align with the trope

2. CONSIDER PLOT ELEMENTS AND NARRATIVE STRUCTURE:
   - Key scenes or moments where the trope might appear
   - Plot developments that typically accompany the trope
   - Narrative patterns associated with the trope

3. EXAMINE THEMATIC ELEMENTS:
   - Themes that often accompany the trope
   - Symbolic representations of the trope
   - Moral or philosophical dimensions of the trope

4. LOOK FOR CONTEXTUAL CLUES:
   - Setting details that might support the trope
   - Historical or cultural contexts relevant to the trope
   - Genre conventions that might influence how the trope appears

5. ANALYZE DIALOGUE AND LANGUAGE:
   - Specific phrases or expressions associated with the trope
   - Dialogue patterns that might reveal the trope
   - Narrative voice or perspective shifts related to the trope

6. PRIORITIZE BASED ON INITIAL EVIDENCE:
   - Give higher priority to paths with stronger initial evidence
   - Consider the specificity and relevance of the evidence
   - Balance breadth of investigation with depth in promising areas

Your investigation plan should be comprehensive, covering different aspects of how the trope might manifest in the book, while prioritizing the most promising paths based on the initial findings.
""",
                param_type=adal.ParameterType.PROMPT,
                requires_opt=True,
            ),
        }

        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=plan_template,
            prompt_kwargs=prompt_params,
            output_processors=parser,
        )


###############################################################################
# Generator for Path Investigation
###############################################################################
@trace_generator_states()
@trace_generator_call(error_only=False)
class PathInvestigationGenerator(adal.Generator):
    def __init__(self, model_client=None, model_kwargs=None):
        investigation_template = r"""
{{task_desc}}
<output_format>
{{output_format_str}}
Example output: 
{
    "path_id": "P1",
    "conclusion": "inconclusive",
    "confidence": 2,
    "reasoning": "Somthing new clues",
    "evidence": [
        "lit20_20_1",
        "lit20_2_20",
    ],
    "follow_up_paths": [
       {
        "id": "P2",
        "description": "Somthing about new directiondescrtion",
        "reasoning": "Somthing about new directionreasoning",
        "search_queries": [
            "Somthing about new directionsearch_queries",
            "Somthing about new directionsearch_queries"
        ],
        "priority": 3,
        "status": "pending"
        },
        {...},
        {...}
    ]
}
</output_format>
<trope_query>
{{query}}
</trope_query>
<investigation_path>
{{investigation_path}}
</investigation_path>
<evidence>
{{evidence}}
</evidence>
<investigation_history>
{{investigation_history}}
</investigation_history>
"""
        parser = adal.DataClassParser(
            data_class=PathInvestigationResult,
            format_type="json",
            return_data_class=True,
        )
        prompt_params = {
            "output_format_str": parser.get_output_format_str(),
            "task_desc": adal.Parameter(
                data="""You are a literary detective analyzing evidence to determine whether a specific narrative trope is present in a book. Your task is to investigate a specific path and draw conclusions based on the evidence provided.

INSTRUCTIONS:
1. ANALYZE the trope query and investigation path to understand what you're looking for.
2. EXAMINE the evidence carefully, looking for indicators of the trope's presence or absence (explicit, implicit, circumstantial, etc.).
3. DRAW a conclusion about whether this path provides evidence for the trope:
   - 'supports' - The evidence supports the presence of the trope
   - 'contradicts' - The evidence contradicts the presence of the trope
   - 'inconclusive' - The evidence is insufficient to determine either way
4. RATE your confidence in the conclusion on a scale of 1-5 (with 5 being highest).
5. PROVIDE detailed reasoning for your conclusion, citing specific evidence.
6. IDENTIFY the most relevant citation IDs that support your conclusion.
7. SUGGEST any follow-up investigation paths if needed.

EFFECTIVE EVIDENCE ANALYSIS STRATEGIES:

1. LOOK FOR EXPLICIT MENTIONS:
   - Direct references to the trope or its components
   - Character statements or narrator descriptions that align with the trope
   - Scenes that clearly demonstrate the trope in action

2. IDENTIFY IMPLICIT INDICATORS:
   - Patterns of behavior or events that suggest the trope
   - Symbolic representations of the trope
   - Thematic elements that align with the trope

3. CONSIDER CONTEXT AND NUANCE:
   - How the evidence fits within the broader narrative
   - Cultural or historical contexts that might affect interpretation
   - Author's style and typical use of literary devices

4. EVALUATE CONTRADICTORY EVIDENCE:
   - Evidence that seems to contradict the presence of the trope
   - Alternative interpretations of the evidence
   - Subversions or inversions of the trope

5. ASSESS EVIDENCE QUALITY:
   - Relevance of the evidence to the specific trope
   - Clarity and directness of the evidence
   - Consistency with other evidence

Your analysis should be thorough, balanced, and focused on the specific investigation path. Cite specific evidence using citation IDs and explain how each piece of evidence supports your conclusion.
""",
                param_type=adal.ParameterType.PROMPT,
                requires_opt=True,
            ),
        }

        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=investigation_template,
            prompt_kwargs=prompt_params,
            output_processors=parser,
        )


###############################################################################
# Generator for Final Trope Answer
###############################################################################
@trace_generator_states()
@trace_generator_call(error_only=False)
class FinalTropeAnswerGenerator(adal.Generator):
    def __init__(self, model_client=None, model_kwargs=None):
        answer_template = r"""
{{task_desc}}
<output_format>
{{output_format_str}}
</output_format>
<trope_query>
{{query}}
</trope_query>
<investigation_results>
{{investigation_results}}
</investigation_results>
<all_evidence>
{{all_evidence}}
</all_evidence>
"""
        parser = adal.DataClassParser(
            data_class=FinalTropeAnswer,
            format_type="json",
            return_data_class=True,
        )
        prompt_params = {
            "output_format_str": parser.get_output_format_str(),
            "task_desc": adal.Parameter(
                data="""You are a literary detective synthesizing evidence to determine whether a specific narrative trope is present in a book. Your task is to provide a final answer based on all the investigation paths that have been explored.

INSTRUCTIONS:
1. ANALYZE the trope query to understand exactly what trope is being investigated.
2. REVIEW all investigation results carefully, considering the conclusions and confidence levels.
3. EXAMINE the evidence cited across all investigations.
4. SYNTHESIZE the findings into a coherent final answer:
   - 'Yes' - The trope is present in the book
   - 'No' - The trope is not present in the book
   - 'Inconclusive' - There is insufficient evidence to determine either way
5. RATE your confidence in the final answer on a scale of 1-5 (with 5 being highest).
6. PROVIDE detailed reasoning for your final answer, explaining how you weighed different evidence.
7. IDENTIFY the most important pieces of evidence that support your conclusion.

EFFECTIVE SYNTHESIS STRATEGIES:

1. WEIGH EVIDENCE QUALITY:
   - Prioritize direct, explicit evidence and implicit or circumstantial evidence
   - Consider the relevance and specificity of each piece of evidence
   - Evaluate the context and significance of each citation

2. BALANCE CONTRADICTORY FINDINGS:
   - Address contradictions between different investigation paths
   - Explain why some evidence might be more compelling than others
   - Consider alternative interpretations of the evidence

3. ASSESS CONFIDENCE HOLISTICALLY:
   - Consider the confidence levels across all investigations
   - Evaluate the breadth and depth of the evidence
   - Acknowledge limitations and uncertainties

4. CONSIDER TROPE VARIATIONS:
   - Recognize that tropes can manifest in different ways
   - Assess whether the evidence supports a variation of the trope
   - Consider subversions or inversions of the trope

5. MAINTAIN LITERARY CONTEXT:
   - Consider the genre, time period, and author's style
   - Evaluate how the trope fits within the broader narrative
   - Assess whether the trope is central or peripheral to the story

Your final answer should be well-reasoned, balanced, and supported by the strongest evidence from across all investigations. Be honest about the limitations of the evidence and the confidence in your conclusion.
""",
                param_type=adal.ParameterType.PROMPT,
                requires_opt=True,
            ),
        }

        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=answer_template,
            prompt_kwargs=prompt_params,
            output_processors=parser,
        )


class NarrativeTropeRAG(adal.GradComponent):
    def __init__(
        self,
        desc: str,
        model_client=None,
        model_kwargs=None,
        passages_per_query: int = 20,
        max_iterations: int = 20,
        context_window: int = 3,
    ):
        super().__init__(desc=desc)
        self.passages_per_query = passages_per_query
        self.max_iterations = max_iterations
        self.context_window = context_window
        self.model_client = model_client
        self.model_kwargs = model_kwargs
        # Initialize components
        # self.embedder = M3Embedder(model_name="BAAI/bge-m3", device="cuda:0")
        # self.embedder = adal.Embedder(
        #     model_client=adal.OllamaClient(),
        #     model_kwargs={"model": "bge-m3:latest"} #
        # )
        self.embedder = FastAPIClientEmbedder(base_url="http://localhost:8000")

        # Configure hybrid retriever
        # self.retriever = FAISSRetriever(
        #     top_k=self.passages_per_query, embedder=self.embedder, dimensions=1024
        # )

        # Initialize all generators
        self.query_rephrase_generator = QueryRephraseGenerator(
            model_client=model_client, model_kwargs=model_kwargs
        )
        self.investigation_plan_generator = InvestigationPlanGenerator(
            model_client=model_client, model_kwargs=model_kwargs
        )
        self.path_investigation_generator = PathInvestigationGenerator(
            model_client=model_client, model_kwargs=model_kwargs
        )
        self.final_answer_generator = FinalTropeAnswerGenerator(
            model_client=model_client, model_kwargs=model_kwargs
        )

        # Initialize stores
        self.citation_store = CitationStore()

    def batch_ollama_embed(self, inputs: List[str]) -> List[list[float]]:
        outputs = []
        for input in inputs:
            try:
                output = self.embedder(input)
                outputs.append(output.data[0].embedding)
            except Exception as e:
                logger.error(f"Error embedding input: {input}")
                logger.error(f"Error: {e}")
                continue
        return outputs
    
    def retrieve_string_queries(self, retriever: adal.Retriever, inputs: List[str]) -> List[Document]:
        batch_embedings = self.batch_ollama_embed(inputs)
        return retriever.retrieve_embedding_queries(batch_embedings)
    
    def _process_documents(self, book_chunks: List[Dict]) -> List[Document]:
        """Enhanced document processing with semantic enrichment"""
        return [
            Document(
                id=chunk["citation_id"],
                text=chunk["text"],
                meta_data={
                    **chunk.get("meta", {}),
                },
                vector=chunk.get("embedding"),
            )
            for chunk in book_chunks
        ]

    def _docs_to_string(self, docs: List[Document]) -> str:
        return "\n".join([f"Citation ID: {doc.id}\nText: {doc.text}" for doc in docs])

    def call(
        self, query: str, book_chunks: List[Dict], id: str | None = None
    ) -> adal.GeneratorOutput:
        """
        Execute the narrative trope investigation pipeline.

        The pipeline follows these steps:
        1. Generate rephrased queries for better semantic search
        2. Retrieve relevant passages for all rephrased queries
        3. Generate an investigation plan with multiple paths
        4. Investigate each path in priority order
        5. Synthesize findings into a final answer
        """
        # Initialize retriever and process documents
        
        retriever = FAISSRetriever(
            top_k=self.passages_per_query, embedder=self.embedder, dimensions=1024
        )
        logger.info(f"Processing {len(book_chunks)} documents")
        docs = self._process_documents(book_chunks)
        logger.info(f"Processed {len(docs)} documents")
        retriever.build_index_from_documents(
            docs, document_map_func=lambda x: x.vector
        )

        logger.info(
            "Step 1: Generating rephrased queries for better semantic search..."
        )
        rephrased_queries_response = rate_limited_call(
            "google_genai",
            self.query_rephrase_generator,
            prompt_kwargs={"query": query},
            id=id
        )

        rephrased_queries = rephrased_queries_response.data.rephrased_queries
        rephrasing_reasoning = rephrased_queries_response.data.reasoning

        logger.info(f"Generated {len(rephrased_queries)} rephrased queries")
        logger.info(f"Rephrasing reasoning: {rephrasing_reasoning}")

        logger.info(
            "Step 2: Retrieving relevant passages for all rephrased queries..."
        )
        all_queries = [query] + rephrased_queries

        all_retrieved_doc_ids = set()
        all_retrieved_docs = []
        query_to_docs = {}

        for q in all_queries:
            try:
                retrieved_context = retriever.retrieve_string_queries(q)
            except Exception as e:
                logger.error(f"Query cause error: {q}")
                logger.error(f"Error retrieving context for query: {e}")
                continue    
            retrieved_context = retrieved_context[0]
           
            query_docs = []
            for i in retrieved_context.doc_indices:
                if i >= len(docs):
                    logger.error(f"Document index {i} is out of range for query. {len(docs)}")
                    continue
                doc = docs[i]
                if doc.id not in all_retrieved_doc_ids:
                    all_retrieved_doc_ids.add(doc.id)
                    query_docs.append(doc)
                    all_retrieved_docs.append(doc)
            query_to_docs[q] = query_docs

            if query_docs:
                self.citation_store.add_citations(
                    [doc.id for doc in query_docs], [doc.text for doc in query_docs]
                )
                logger.info(
                    f"Found {len(query_docs)} new relevant passages for query: {q}"
                )

        logger.info(f"Total unique documents retrieved: {len(all_retrieved_docs)}")

        if not all_retrieved_docs:
            logger.warning("No relevant documents found for any query.")
            final_answer = FinalTropeAnswer(
                answer="No",
                confidence=5,
                reasoning="No relevant passages were found in the book for any of the queries related to this trope.",
                key_evidence=[],
            )
            return adal.GeneratorOutput(id=id, data=final_answer)

        initial_findings = []
        for q, q_docs in query_to_docs.items():
            if q_docs:
                doc_summary = "\n".join(
                    [f"Citation ID: {doc.id}\nText: {doc.text}" for doc in q_docs]
                )
                initial_findings.append(
                    f"Query: {q}\nDocuments found: {len(q_docs)}\n{doc_summary}"
                )
            else:
                initial_findings.append(f"Query: {q}\nNo documents found.")

        initial_findings_str = "\n\n".join(initial_findings)

        logger.info("Step 3: Generating investigation plan...")
        investigation_plan_response = rate_limited_call(
            "google_genai",
            self.investigation_plan_generator,
            prompt_kwargs={
                "query": query,
                "initial_findings": initial_findings_str,
            },
            id=id
        )

        investigation_plan = investigation_plan_response.data
        investigation_paths = sorted(
            investigation_plan.paths, key=lambda p: p.priority
        )

        logger.info(
            f"Generated investigation plan with {len(investigation_paths)} paths"
        )
        logger.info(f"Investigation plan reasoning: {investigation_plan.reasoning}")

        logger.info("Step 4: Investigating paths in priority order...")

        investigation_results = []
        investigation_history = []

        current_iteration = 0

        for path_index, path in enumerate(investigation_paths):
            logger.info(
                f"Investigating path {path_index + 1}/{len(investigation_paths)}: {path.description}"
            )
            if current_iteration > self.max_iterations:
                logger.info(f"Max iterations reached. Stopping investigation.")
                break
            current_iteration += 1
            path.status = "in_progress"

            path_docs = []
            path_doc_ids = set()

            for search_query in path.search_queries:
                logger.info(
                    f"Retrieving documents for search query: {search_query}"
                )
                retrieved_context = retriever.retrieve_string_queries(search_query)
                if not retrieved_context:
                    logger.error(f"Retrieved context is empty for search query.")
                    continue
                retrieved_context = retrieved_context[0]

                for i in retrieved_context.doc_indices:
                    doc = docs[i]
                    if doc.id not in path_doc_ids:
                        path_doc_ids.add(doc.id)
                        path_docs.append(doc)

                self.citation_store.add_citations(
                    [
                        doc.id
                        for doc in path_docs
                        if doc.id not in all_retrieved_doc_ids
                    ],
                    [
                        doc.text
                        for doc in path_docs
                        if doc.id not in all_retrieved_doc_ids
                    ],
                )

                for doc in path_docs:
                    if doc.id not in all_retrieved_doc_ids:
                        all_retrieved_doc_ids.add(doc.id)
                        all_retrieved_docs.append(doc)

            logger.info(
                f"Found {len(path_docs)} documents for path: {path.description}"
            )

            if not path_docs:
                logger.warning(f"No documents found for path: {path.description}")
                path.status = "completed"

                path_result = PathInvestigationResult(
                    path_id=path.id,
                    conclusion="inconclusive",
                    confidence=1,
                    reasoning=f"No relevant documents found for this investigation path.",
                    evidence=[],
                    follow_up_paths=[],
                )

                investigation_results.append(path_result)
                investigation_history.append(
                    f"Path {path_index + 1}: {path.description} - No documents found"
                )
                continue

            evidence_str = self._docs_to_string(path_docs)

            investigation_history_str = "\n\n".join(investigation_history)

            path_investigation_response = rate_limited_call(
                "google_genai",
                self.path_investigation_generator,
                prompt_kwargs={
                    "query": query,
                    "investigation_path": json.dumps(
                        path, default=lambda o: o.__dict__
                    ),
                    "evidence": evidence_str,
                    "investigation_history": investigation_history_str,
                },
                id=id
            )
            if path_investigation_response.data is None:
                logger.error(f"Path investigation response is None for path: {path.description}")
                continue
            path_result = path_investigation_response.data
            path.status = "completed"
            current_iteration += 1
            investigation_results.append(path_result)

            investigation_history.append(
                f"Path {path_index + 1}: {path.description}\n"
                f"Conclusion: {path_result.conclusion}\n"
                f"Confidence: {path_result.confidence}/5\n"
                f"Reasoning: {path_result.reasoning}\n"
                f"Evidence: {', '.join(path_result.evidence)}"
            )

            if path_result.follow_up_paths:
                logger.info(
                    f"Adding {len(path_result.follow_up_paths)} follow-up paths to investigation"
                )
                for follow_up_path in path_result.follow_up_paths:
                    follow_up_path.priority = (
                        len(investigation_paths) + 1
                    )  # Lower priority than original paths
                    investigation_paths.append(follow_up_path)

        logger.info("Step 5: Synthesizing findings into final answer...")

        investigation_results_str = json.dumps(
            [result.__dict__ for result in investigation_results],
            default=lambda o: o.__dict__,
        )

        all_evidence_str = self._docs_to_string(all_retrieved_docs)

        final_answer_response = rate_limited_call(
            "google_genai",
            self.final_answer_generator,
            prompt_kwargs={
                "query": query,
                "investigation_results": investigation_results_str,
                "all_evidence": all_evidence_str,
            },
            id=id
        )

        final_answer = final_answer_response.data

        logger.info(f"Final answer: {final_answer.answer}")
        logger.info(f"Confidence: {final_answer.confidence}/5")
        logger.info(f"Reasoning: {final_answer.reasoning}")

        return adal.GeneratorOutput(id=id, data=final_answer)

    def forward(
        self, query: str, book_chunks: List[Dict[str, str]], id: str | None = None
    ) -> adal.Parameter:
        return self.call(query, book_chunks, id)


class LLMJudgeGenerator(adal.Generator):
    def __init__(self, model_client=None, model_kwargs=None):
        judge_template = r"""
{{task_desc}}

<trope_query>
{{query}}
</trope_query>

{%if ground_truth%}
<ground_truth>
{{ground_truth}}
</ground_truth>
{%endif%}

{%if evidence_trace%}
<evidence_trace>
{{evidence_trace}}
</evidence_trace>
{%endif%}

<model_answer>
{{model_answer}}
</model_answer>

Please evaluate the model's answer and provide a score between 0 and 1, where:
- 0 means the answer is completely incorrect or irrelevant
- 1 means the answer is completely correct and addresses all aspects of the query

Your response should be a single number between 0 and 1.
"""
        prompt_params = {
            "task_desc": adal.Parameter(
                data="""You are an expert literary judge evaluating answers about narrative tropes in literature. Your task is to assess how well a model's answer addresses a query about whether a specific trope appears in a book.

EVALUATION CRITERIA:
1. CORRECTNESS: Does the answer correctly identify the presence or absence of the trope?
2. EVIDENCE: Does the answer cite relevant evidence from the text?
3. REASONING: Does the answer provide sound reasoning that connects the evidence to the conclusion?
4. COMPREHENSIVENESS: Does the answer address all key aspects of the trope mentioned in the query?
5. NUANCE: Does the answer acknowledge complexity, partial matches, or ambiguity when appropriate?

You will be provided with:
1. The original query about a narrative trope
2. The ground truth answer
3. The evidence trace that led to the model's answer
4. The model's answer to be evaluated


Your task is to evaluate the model's answer against the ground truth and provide a score between 0 and 1, where 0 is completely incorrect and 1 is perfectly correct.
Your response should be a single number between 0 and 1.
""",
                param_type=adal.ParameterType.PROMPT,
                requires_opt=True,
            ),
        }

        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=judge_template,
            prompt_kwargs=prompt_params,
            output_processors=adal.FloatParser(),
        )


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
        judge_model_client: adal.ModelClient,
        judge_model_kwargs: Dict,
        backward_engine_model_config: Dict | None = None,
        teacher_model_config: Dict | None = None,
        text_optimizer_model_config: Dict | None = None,
        passages_per_query: int = 5,
    ):
        task = NarrativeTropeRAG(
            desc="Narrative Trope RAG Pipeline",
            model_client=model_client,
            model_kwargs=model_kwargs,
            passages_per_query=passages_per_query,
        )

        # self.llm_judge = LLMJudgeGenerator(
        #     model_client=judge_model_client, model_kwargs=judge_model_kwargs
        # )

        # def eval_fn(y, y_gt):
        #     output = self.llm_judge.call(
        #         prompt_kwargs={
        #             "query": y,
        #             "ground_truth": y_gt,
        #             "evidence_trace": "",
        #             "model_answer": "",
        #         }
        #     )
        #     return output.data

        def eval_fn(y, y_gt):
            return 1 if y == y_gt else 0

        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="LLM Judge Evaluation score between 0 and 1 for the model's answer.",
        )
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            teacher_model_config=teacher_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
        )

    def prepare_task(self, sample: TropeQuery) -> Tuple[Callable[..., Any], Dict]:
        # Sample is expected to contain "query", "book_chunks", "answer", and optionally "id".
        if self.task.training:
            return self.task.forward, {
                "query": sample.query,
                "book_chunks": load_jsonl(
                    f"preprocessed_books/{sample.book_id}.jsonl"
                ),
                "id": sample.id,
            }
        else:
            return self.task.call, {
                "query": sample.query,
                "book_chunks": load_jsonl(
                    f"preprocessed_books/{sample.book_id}.jsonl"
                ),
                "id": sample.id,
            }

    def prepare_eval(
        self, sample: TropeQuery, y_pred: adal.GeneratorOutput
    ) -> float:
        if y_pred.data is not None:
            y_label = y_pred.data.answer
            if "yes" in y_label.lower():
                y_label = True
            else:
                y_label = False
            return self.eval_fn, {"y": y_label, "y_gt": sample.answer}
        else:
            return self.eval_fn, {"y": False, "y_gt": True}

    def prepare_loss(self, sample: TropeQuery, pred: adal.Parameter):
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.answer,
            eval_input=sample.answer,
            requires_opt=False,
        )
        pred.eval_input = pred.answer
        return self.loss_fn, {"y": pred, "y_gt": y_gt}


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

    # model_client = adal.OllamaClient(host="http://localhost:11434") .
    # model_kwargs = {
    #     "model": "qwen2.5:14b-instruct-q8_0"
    # }

    model_client = adal.GoogleGenAIClient()
    model_kwargs = {
        "model": "gemini-2.0-flash",
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    adal_component = NarrativeTropeAdal(
        model_client=model_client, model_kwargs=model_kwargs, passages_per_query=5
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
    book_chunks = load_jsonl(f"{book_id}.jsonl")
    # book_chunks = process_epub(f"{book_id}.epub", book_id, max_chunk_length=100)
    user_query = """Does the book contain this trope - It Has Been an Honor - definition: Honor, loyalty, camaraderie. Some people swear by it, others quietly live by it, many ignore it altogether. When you are ready to say your Last Words to a friend or ally, what would be the most appropriate words to say between two Proud Warrior Race Guys, or, really, anyone who's feeling honorable and doomed at the moment? "It has been an honor... to fight beside you." Often followed by a response like: "The honor has been mine."
There is a simplicity in this statement that you rarely find; saying this will usually be the only thing you need to say, then you can make your Heroic Sacrifice or face your Bolivian Army Ending. Of course, this statement is frequently followed by the arrival of The Cavalry. Often a part of a Pre-Sacrifice Final Goodbye. Sometimes spoken to an enemy by the Worthy Opponent. May be the demonstration of Fire Forged Friendship, even if they left it to the last minute to acknowledge it.
The sentiment can be implied rather than spoken directly, through Something They Would Never Say, O.O.C. Is Serious Business, a handshake, salute or even a Precision F-Strike. Often a cause of Manly Tears, both for the characters involved and the audience. It may also be given a little gallows humour through more flippant variants such as 'It's been nice knowing you', which can be played either for comedic irony or poignancy depending on the source.
Note that only one character has to be certain of death for this trope; in the face of an impending Heroic Sacrifice (where More Hero than Thou is clearly ridiculous), the other characters may say it to the sacrificing hero, or the hero may say it to those who will survive. However, there is a chance that any hero who says those kinds of words might end up surviving the ordeal if Deus ex Machina comes into play such as getting outside help from someone else.
May feature in the Final Speech.
Compare So Proud of You, I Die Free, Do Not Go Gentle, Face Death with Dignity, Obi-Wan Moment. One way to subvert this is to set up for It Has Been An Honor and have the character deliver a Dying Declaration of Hate instead. See also Last Moment Together.
"""
    test_narrative_trope_pipeline(book_chunks=book_chunks, user_query=user_query)




def load_dataset(cache_dir: str, n: int = None) -> List[TropeQuery]:
    """
    Dataset consists of
    - book_id
    - split (train, val, test)
    - trope_name
    - trope_definition
    - trope_example

    where trope_example is a ground truth example of the trope in the book, if None, then the trope is not present in the book.
    """
    import pandas as pd

    df = pd.read_csv("tropes_dataset.csv")
    df["query"] = (
        "Does the book contain the trope "
        + df["trope_name"]
        + " - "
        + df["trope_definition"]
    )
    df["answer"] = df["trope_example"].notna()
    df = df.sample(n=n)
    return [
        TropeQuery(
            id=row["id"],
            book_id=row["book_id"],
            trope_name=row["trope_name"],
            trope_definition=row["trope_definition"],
            trope_example=row["trope_example"],
            query=row["query"],
            answer=row["answer"],
        )
        for row in df.to_dict(orient="records")
    ]


def train():
    model_client = adal.GoogleGenAIClient()
    model_kwargs = {
        "model": "gemini-2.0-flash",
        "temperature": 0.7,
        "top_p": 0.95,
    }

    judge_model_client = adal.GoogleGenAIClient()
    judge_model_kwargs = {
        "model": "gemini-2.0-flash",
        "temperature": 0.7,
        "top_p": 0.95,
    }

    backward_engine_model_config = {
        "model_client": adal.GoogleGenAIClient,
        "model_kwargs": model_kwargs,
    }

    teacher_model_config = {
        "model_client": adal.GoogleGenAIClient,
        "model_kwargs": model_kwargs,
    }

    text_optimizer_model_config = {
        "model_client": adal.GoogleGenAIClient,
        "model_kwargs": model_kwargs,
    }

    adal_component = NarrativeTropeAdal(
        model_client=model_client,
        model_kwargs=model_kwargs,
        passages_per_query=5,
        judge_model_client=judge_model_client,
        judge_model_kwargs=judge_model_kwargs,
        backward_engine_model_config=backward_engine_model_config,
        teacher_model_config=teacher_model_config,
        text_optimizer_model_config=text_optimizer_model_config,
    )

    trainset = load_dataset(cache_dir="preprocessed_books", n=100)
    trainer = adal.Trainer(
        adaltask=adal_component,
        ckpt_path="checkpoints/",
        train_batch_size=4
    )
    trainer.diagnose(dataset=trainset, split="train")


if __name__ == "__main__":
    # test_on_book("lit20")
    train()
