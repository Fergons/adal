import os
import orjson
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import adalflow as adal
from adalflow.core.types import DataClass, Document
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from m3_embedder import M3Embedder
from process_ebooks import extract_text_from_epub, load_jsonl, process_epub
from rate_limiter import rate_limited_call
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import dotenv
dotenv.load_dotenv()

from opik import Opik

opik_client = Opik(project_name="tvtropes_gen", api_key=os.getenv("OPIK_API_KEY"))


@dataclass
class TropeEvidence(DataClass):
    """Evidence supporting a trope in a book."""
    citation_ids: List[str] = field(
        default_factory=list,
        metadata={
            "description": "List of citation ids (e.g. lit1_132, lit20_243) from the relevant passages that support the trope example provided by the user."
        },
    )
    paragraphs: List[str] = field(
        default_factory=list,
        metadata={
            "description": "List of paragraphs from the relevant passages that support the trope example provided by the user."
        },
    )
    reasoning: str = field(
        default_factory=str,
        metadata={
            "description": "Reasoning explaining how these paragraphs support the trope example provided by the user."
        },
    )
    __input_fields__ = []
    __output_fields__ = ["citation_ids", "reasoning"]


class TropeEvidenceGenerator(adal.Generator):
    """Generator for finding evidence supporting a trope in a book."""
    
    def __init__(self, model_client=None, model_kwargs=None):
        evidence_template = r"""
{{task_desc}}
<output_format>
{{output_format_str}}
</output_format>
<trope_description>
{{trope_description}}
</trope_description>
<user_example>
{{user_example}}
</user_example>
<relevant_passages>
{{relevant_passages}}
</relevant_passages>
"""
        parser = adal.DataClassParser(
            data_class=TropeEvidence,
            format_type="json",
            return_data_class=True,
        )
        prompt_params = {
            "output_format_str": parser.get_output_format_str(),
            "task_desc": adal.Parameter(
                data="""You are a literary analyst specialized in identifying narrative tropes in books. Your task is to analyze relevant passages from a book and identify the strongest evidence supporting a specific trope example provided by the user.

INSTRUCTIONS:
1. ANALYZE the trope description and the user's example of how this trope appears in the book.
2. EXAMINE the relevant paragraphs to find the most relevant passages that support the trope.
3. SELECT all paragraphs that explicitly or implicitly demonstrate the trope or explain a part of the reasoning and related facts.
4. RETURN the citation ids of all the relevant passages and the paragraphs that support the trope.

SELECTION CRITERIA:
1. RELEVANCE: How directly the passage demonstrates the trope
2. CLARITY: How clearly the trope elements are presented
3. SIGNIFICANCE: How important the passage is to understanding the trope's presence
4. CONTEXT: How the passage fits within the broader narrative

PROVIDE REASONING:
Explain your overall analysis of how these passages collectively support the user's example of the trope. Consider:
- Patterns across multiple passages
- Character development or relationships that demonstrate the trope
- Plot elements that embody the trope
- Thematic connections to the trope

Your analysis should be thorough, focused on textual evidence, and directly connected to the specific trope example provided by the user.
""",
                param_type=adal.ParameterType.PROMPT,
                requires_opt=True,
            ),
        }

        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=evidence_template,
            prompt_kwargs=prompt_params,
            output_processors=parser,
        )


class SimpleTVTropesRAG:
    """Simple RAG pipeline for finding evidence of tropes in books."""
    
    def __init__(
        self,
        model_client=None,
        model_kwargs=None,
        passages_per_query: int = 15,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
    ):
        self.passages_per_query = passages_per_query
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_client = model_client
        self.model_kwargs = model_kwargs
        
        # Initialize components
        self.embedder = M3Embedder(model_name="BAAI/bge-m3", device="cuda:0")
        
        # Configure retriever
        self.retriever = FAISSRetriever(
            top_k=self.passages_per_query, 
            embedder=self.embedder, 
            dimensions=1024
        )
        
        # Initialize generator
        self.evidence_generator = TropeEvidenceGenerator(
            model_client=model_client, 
            model_kwargs=model_kwargs
        )
    
    def _docs_to_string(self, docs: List[Document]) -> str:
        """Convert documents to a formatted string."""
        return "\n\n".join([f"Document with id {doc.id}:\n{doc.text}" for doc in docs])
    
    def _chunks_to_docs(self, book_chunks: List[Dict]) -> List[Document]:
        """Enhanced document processing with semantic enrichment"""
        return [
            Document(
                id=chunk["citation"],
                text=chunk["text"],
                meta_data={
                    **chunk.get("meta", {}),
                },
                vector=chunk.get("embedding"),
            )
            for chunk in book_chunks
        ]

    def find_trope_evidence(
        self,
        book_text: str,
        trope_description: str, 
        user_example: str,
        book_chunks: List[Dict]
    ) -> TropeEvidence:
        """
        Find evidence in the book supporting the user's trope example.
        
        Args:
            book_text: The full text of the book
            trope_description: Description of the trope
            user_example: User's explanation of how the trope appears in the book
            book_chunks: List of book chunks
            
        Returns:
            TropeEvidence object containing relevant paragraphs and reasoning
        """
        logger.info("Processing book text...")
        docs = self._chunks_to_docs(book_chunks)

        # Reset and build index
        self.retriever.reset_index()
        self.retriever.build_index_from_documents(docs, document_map_func=lambda x: x.vector)
        
        combined_query = f"{trope_description}\n\n{user_example}"
        
        logger.info("Retrieving relevant passages...")
        retrieved_contexts = self.retriever.retrieve_string_queries(combined_query)
        retrieved_context = retrieved_contexts[0]
        
        # Get relevant documents
        relevant_docs = [docs[i] for i in retrieved_context.doc_indices]
        
        logger.info(f"Found {len(relevant_docs)} relevant passages")
        
        relevant_passages = self._docs_to_string(relevant_docs)
        
        logger.info("Generating evidence analysis...")
        evidence_response = rate_limited_call(
            "google_genai",
            self.evidence_generator,
            prompt_kwargs={
                "trope_description": trope_description,
                "user_example": user_example,
                "relevant_passages": relevant_passages
            }
        )
        paragraphs = [doc.text for doc in docs if doc.id in evidence_response.data.citation_ids]
        evidence_response.data.paragraphs = paragraphs
        return evidence_response.data



def test_simple_tvtropes(
    book_path: str, 
    trope_description: str, 
    user_example: str,
    passages_per_query: int = 100,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """Test the SimpleTVTropesRAG pipeline with a book file."""
    # Read book text
    logger.info(f"Reading book from {book_path}...")
    
    # Check if it's an EPUB file
    if not book_path.lower().endswith('.epub'):
        return None
    
    book_chunks = load_jsonl(book_path.replace(".epub", ".jsonl"))
    book_text = "\n\n".join([f"Document with id {chunk['citation']}:\n{chunk['text']}" for chunk in book_chunks])

    # Initialize model client
    # model_client = adal.GoogleGenAIClient()
    # model_kwargs = {
    #     "model": "gemini-2.0-flash",
    #     "temperature": 0.8,
    #     "top_p": 0.95,
    #     "top_k": 40,
    #     "max_output_tokens": 8192,
    #     "response_mime_type": "text/plain",
    # }

    model_client = adal.OpenAIClient(base_url="https://openrouter.ai/api/v1")
    model_kwargs = {
        "model": "qwen/qwq-32b:free",
        "temperature": 1.0,
        "top_p": 1.0,
    }

    # Create pipeline
    pipeline = SimpleTVTropesRAG(
        model_client=model_client,
        model_kwargs=model_kwargs,
        passages_per_query=passages_per_query,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Find evidence
    evidence = pipeline.find_trope_evidence(
        book_text=book_text,
        trope_description=trope_description,
        user_example=user_example,
        book_chunks=book_chunks
    )

    # Print results
    print("\n=== TROPE EVIDENCE ===\n")
    print(f"Trope: {trope_description}")
    print(f"User Example: {user_example}")
    print("\nEvidence Paragraphs:")
    for paragraph in evidence.paragraphs:
        print(paragraph)
    print("\nReasoning:")
    print(evidence.reasoning)
    return evidence


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find evidence of tropes in books')
    parser.add_argument('--book', type=str, help='Path to the book file (EPUB or text)')
    parser.add_argument('--trope', type=str, help='Description of the trope to find')
    parser.add_argument('--example', type=str, help='User example of how the trope appears in the book')
    parser.add_argument('--output', type=str, help='Path to save the results as JSON (optional)')
    parser.add_argument('--passages', type=int, default=20, help='Number of passages to retrieve per query (default: 15)')
    parser.add_argument('--chunk-size', type=int, default=200, help='Size of text chunks (default: 1000)')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='Overlap between chunks (default: 200)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Example usage if no arguments provided
    if not args.book or not args.trope or not args.example:
        # Default example
        trope_description = """Does the book contain this trope - It Has Been an Honor - definition: Honor, loyalty, camaraderie. Some people swear by it, others quietly live by it, many ignore it altogether. When you are ready to say your Last Words to a friend or ally, what would be the most appropriate words to say between two Proud Warrior Race Guys, or, really, anyone who's feeling honorable and doomed at the moment? "It has been an honor... to fight beside you." Often followed by a response like: "The honor has been mine."
There is a simplicity in this statement that you rarely find; saying this will usually be the only thing you need to say, then you can make your Heroic Sacrifice or face your Bolivian Army Ending. Of course, this statement is frequently followed by the arrival of The Cavalry. Often a part of a Pre-Sacrifice Final Goodbye. Sometimes spoken to an enemy by the Worthy Opponent. May be the demonstration of Fire Forged Friendship, even if they left it to the last minute to acknowledge it.
The sentiment can be implied rather than spoken directly, through Something They Would Never Say, O.O.C. Is Serious Business, a handshake, salute or even a Precision F-Strike. Often a cause of Manly Tears, both for the characters involved and the audience. It may also be given a little gallows humour through more flippant variants such as 'It's been nice knowing you', which can be played either for comedic irony or poignancy depending on the source.
Note that only one character has to be certain of death for this trope; in the face of an impending Heroic Sacrifice (where More Hero than Thou is clearly ridiculous), the other characters may say it to the sacrificing hero, or the hero may say it to those who will survive. However, there is a chance that any hero who says those kinds of words might end up surviving the ordeal if Deus ex Machina comes into play such as getting outside help from someone else.
May feature in the Final Speech."""
        user_example = "No example provided. You have to reason about the trope and find the evidence."
        book_path = "lit20.epub"  # Replace with actual path
        
        print("No arguments provided, using default example:")
        print(f"Book: {book_path}")
        print(f"Trope: {trope_description}")
        print(f"Example: {user_example}")

    else:
        trope_description = args.trope
        user_example = args.example
        book_path = args.book 
    
    # Run the pipeline
    evidence = test_simple_tvtropes(
        book_path=book_path,
        trope_description=trope_description,
        user_example=user_example,
        passages_per_query=args.passages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Save results if output path provided
    if args.output:
        try:
            # Convert to dictionary
            evidence_dict = {
                "book_id": Path(book_path).stem,
                "citation_ids": evidence.citation_ids,
                "paragraphs": evidence.paragraphs,  
                "reasoning": evidence.reasoning
            }
            
            # Save to JSON
            with open(args.output, 'a', encoding='utf-8') as f:
                f.write(orjson.dumps(evidence_dict))
                f.write("\n")
            
            logger.info(f"Results saved to {args.output}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
