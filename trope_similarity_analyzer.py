#!/usr/bin/env python
"""
Advanced pipeline for analyzing tropes in the context of specific books and extracting evidence for story similarity.
This pipeline combines trope analysis with evidence extraction to identify similar stories based on shared tropes.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import adalflow as adal
from adalflow.core.types import DataClass, Document
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from m3_embedder import M3Embedder
from rate_limiter import rate_limited_call
from process_ebooks import load_jsonl
from trope_analysis import TropeSimilarityPipeline, TropeAnalysisResult, TropeSimilarityCategory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
import dotenv
dotenv.load_dotenv()


@dataclass
class TropeEvidence(DataClass):
    """Evidence supporting a trope in a book."""
    book_id: str = field(
        default_factory=str,
        metadata={
            "description": "Identifier of the book being analyzed."
        },
    )
    trope_name: str = field(
        default_factory=str,
        metadata={
            "description": "Name of the trope being analyzed."
        },
    )
    citation_ids: List[str] = field(
        default_factory=list,
        metadata={
            "description": "List of citation ids from the relevant passages that support the trope."
        },
    )
    paragraphs: List[str] = field(
        default_factory=list,
        metadata={
            "description": "List of paragraphs from the relevant passages that support the trope."
        },
    )
    confidence_score: int = field(
        default=0,
        metadata={
            "description": "Confidence score (1-10) indicating how strongly the evidence supports the presence of the trope."
        },
    )
    reasoning: str = field(
        default_factory=str,
        metadata={
            "description": "Reasoning explaining how these paragraphs support the trope."
        },
    )
    __input_fields__ = ["book_id", "trope_name"]
    __output_fields__ = ["citation_ids", "confidence_score", "reasoning"]


@dataclass
class StorySimilarityResult(DataClass):
    """Result of comparing two stories based on shared tropes."""
    story1_id: str = field(
        default_factory=str,
        metadata={
            "description": "Identifier of the first story."
        },
    )
    story2_id: str = field(
        default_factory=str,
        metadata={
            "description": "Identifier of the second story."
        },
    )
    similarity_score: float = field(
        default=0.0,
        metadata={
            "description": "Overall similarity score (0-1) between the two stories."
        },
    )
    shared_tropes: List[Dict[str, Any]] = field(
        default_factory=list,
        metadata={
            "description": "List of tropes shared by both stories, with relevance scores and evidence."
        },
    )
    unique_tropes_story1: List[str] = field(
        default_factory=list,
        metadata={
            "description": "List of tropes found only in the first story."
        },
    )
    unique_tropes_story2: List[str] = field(
        default_factory=list,
        metadata={
            "description": "List of tropes found only in the second story."
        },
    )
    analysis: str = field(
        default_factory=str,
        metadata={
            "description": "Detailed analysis of the similarity between the two stories."
        },
    )
    __input_fields__ = ["story1_id", "story2_id"]
    __output_fields__ = ["similarity_score", "shared_tropes", "unique_tropes_story1", "unique_tropes_story2", "analysis"]


class TropeEvidenceGenerator(adal.Generator):
    """Generator for finding evidence supporting a trope in a book."""
    
    def __init__(self, model_client=None, model_kwargs=None):
        evidence_template = r"""
{{task_desc}}
<output_format>
{{output_format_str}}
</output_format>
<trope>
Name: {{trope_name}}
Description: {{trope_description}}
Relevance for Story Similarity: {{trope_relevance}}
</trope>
<book_id>
{{book_id}}
</book_id>
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
                data="""You are a literary analyst specialized in identifying narrative tropes in books. Your task is to analyze relevant passages from a book and identify evidence supporting the presence of a specific trope.

INSTRUCTIONS:
1. ANALYZE the trope description and its relevance for story similarity.
2. EXAMINE the relevant paragraphs to find evidence that supports the presence of this trope in the book.
3. SELECT all paragraphs that explicitly or implicitly demonstrate the trope.
4. ASSIGN a confidence score (1-10) indicating how strongly the evidence supports the presence of the trope:
   - 10: Definitive, explicit evidence that perfectly matches the trope
   - 7-9: Strong evidence with clear examples of the trope
   - 4-6: Moderate evidence suggesting the trope is present
   - 1-3: Weak or ambiguous evidence that only hints at the trope

5. PROVIDE detailed reasoning explaining how the selected passages demonstrate the trope.

SELECTION CRITERIA:
1. RELEVANCE: How directly the passage demonstrates the trope
2. CLARITY: How clearly the trope elements are presented
3. SIGNIFICANCE: How important the passage is to understanding the trope's presence
4. CONTEXT: How the passage fits within the broader narrative

Your analysis should be thorough, focused on textual evidence, and directly connected to the specific trope being analyzed.
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


class StorySimilarityGenerator(adal.Generator):
    """Generator for comparing two stories based on shared tropes."""
    
    def __init__(self, model_client=None, model_kwargs=None):
        similarity_template = r"""
{{task_desc}}
<output_format>
{{output_format_str}}
</output_format>
<story1>
ID: {{story1_id}}
Tropes: {{story1_tropes}}
</story1>
<story2>
ID: {{story2_id}}
Tropes: {{story2_tropes}}
</story2>
<shared_trope_evidence>
{{shared_trope_evidence}}
</shared_trope_evidence>
"""
        parser = adal.DataClassParser(
            data_class=StorySimilarityResult,
            format_type="json",
            return_data_class=True,
        )
        prompt_params = {
            "output_format_str": parser.get_output_format_str(),
            "task_desc": adal.Parameter(
                data="""You are a comparative literature expert specializing in analyzing story similarities based on shared narrative tropes. Your task is to compare two stories and assess their similarity based on the tropes they share.

INSTRUCTIONS:
1. ANALYZE the tropes found in each story and the evidence supporting them.
2. IDENTIFY which tropes are shared between the stories and which are unique to each.
3. CALCULATE an overall similarity score (0-1) based on:
   - The number and relevance of shared tropes
   - The strength of evidence for each shared trope
   - The significance of the shared tropes to each story's identity
   - The balance between shared and unique tropes

4. PROVIDE a detailed analysis of the similarity between the two stories, including:
   - How the shared tropes contribute to similarity
   - How the unique tropes differentiate the stories
   - Whether the similarities are superficial or fundamental
   - Whether the stories could be considered variations of the same basic narrative

SCORING GUIDELINES:
- 0.9-1.0: Nearly identical stories with the same fundamental structure and tropes
- 0.7-0.8: Highly similar stories with strong shared narrative elements
- 0.5-0.6: Moderately similar stories with significant shared tropes
- 0.3-0.4: Somewhat similar stories with a few important shared elements
- 0.1-0.2: Minimally similar stories with minor shared tropes
- 0.0: No meaningful similarity between the stories

Your analysis should be nuanced, considering both the quantity and quality of shared tropes, as well as their significance to each story's identity.
""",
                param_type=adal.ParameterType.PROMPT,
                requires_opt=True,
            ),
        }

        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=similarity_template,
            prompt_kwargs=prompt_params,
            output_processors=parser,
        )


class TropeSimilarityAnalyzer:
    """Advanced pipeline for analyzing tropes in books and comparing story similarity."""
    
    def __init__(
        self,
        model_client=None,
        model_kwargs=None,
        passages_per_query: int = 15,
    ):
        self.passages_per_query = passages_per_query
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
        
        # Initialize generators
        self.trope_analyzer = TropeSimilarityPipeline(
            model_client=model_client, 
            model_kwargs=model_kwargs
        )
        
        self.evidence_generator = TropeEvidenceGenerator(
            model_client=model_client, 
            model_kwargs=model_kwargs
        )
        
        self.similarity_generator = StorySimilarityGenerator(
            model_client=model_client, 
            model_kwargs=model_kwargs
        )
        
        # Store for analyzed tropes and evidence
        self.analyzed_tropes = {}
        self.trope_evidence = defaultdict(dict)  # book_id -> trope_name -> evidence
    
    def _docs_to_string(self, docs: List[Document]) -> str:
        """Convert documents to a formatted string."""
        return "\n\n".join([f"Document with id {doc.id}:\n{doc.text}" for doc in docs])
    
    def _chunks_to_docs(self, book_chunks: List[Dict]) -> List[Document]:
        """Convert book chunks to Document objects."""
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
    
    def analyze_tropes(self, tropes: List[Dict[str, str]]) -> Dict[str, TropeAnalysisResult]:
        """
        Analyze a list of tropes for their relevance in story similarity assessment.
        
        Args:
            tropes: List of dictionaries, each containing 'name' and 'description' keys
            
        Returns:
            Dictionary mapping trope names to their analysis results
        """
        logger.info(f"Analyzing {len(tropes)} tropes for story similarity relevance")
        
        # Analyze each trope individually
        for trope in tropes:
            try:
                if trope['name'] not in self.analyzed_tropes:
                    result = self.trope_analyzer.analyze_trope(trope['name'], trope['description'])
                    self.analyzed_tropes[trope['name']] = result
                    logger.info(f"Analyzed trope '{trope['name']}' - Relevance score: {result.similarity_relevance_score}")
            except Exception as e:
                logger.error(f"Error analyzing trope '{trope['name']}': {str(e)}")
        
        return self.analyzed_tropes
    
    def find_trope_evidence(
        self,
        book_id: str,
        trope_name: str,
        trope_description: str,
        book_chunks: List[Dict]
    ) -> TropeEvidence:
        """
        Find evidence in a book supporting a specific trope.
        
        Args:
            book_id: Identifier of the book
            trope_name: Name of the trope
            trope_description: Description of the trope
            book_chunks: List of book chunks
            
        Returns:
            TropeEvidence object containing relevant paragraphs and reasoning
        """
        # Check if we've already analyzed this trope for this book
        if trope_name in self.trope_evidence.get(book_id, {}):
            return self.trope_evidence[book_id][trope_name]
        
        logger.info(f"Finding evidence for trope '{trope_name}' in book '{book_id}'")
        
        # Get trope analysis if available
        trope_analysis = self.analyzed_tropes.get(trope_name)
        trope_relevance = ""
        if trope_analysis:
            trope_relevance = f"""
Similarity Relevance Score: {trope_analysis.similarity_relevance_score}/10
Primary Category: {trope_analysis.primary_category}
Secondary Categories: {', '.join(trope_analysis.secondary_categories) if trope_analysis.secondary_categories else 'None'}
Reasoning: {trope_analysis.reasoning}
"""
        
        # Process book chunks
        docs = self._chunks_to_docs(book_chunks)
        
        # Reset and build index
        self.retriever.reset_index()
        self.retriever.build_index_from_documents(docs, document_map_func=lambda x: x.vector)
        
        # Create query from trope description
        query = f"{trope_name}: {trope_description}"
        
        # Retrieve relevant passages
        retrieved_contexts = self.retriever.retrieve_string_queries(query)
        retrieved_context = retrieved_contexts[0]
        
        # Get relevant documents
        relevant_docs = [docs[i] for i in retrieved_context.doc_indices]
        
        logger.info(f"Found {len(relevant_docs)} relevant passages for trope '{trope_name}' in book '{book_id}'")
        
        # If no relevant passages found, return minimal evidence
        if not relevant_docs:
            evidence = TropeEvidence(
                book_id=book_id,
                trope_name=trope_name,
                citation_ids=[],
                paragraphs=[],
                confidence_score=0,
                reasoning="No relevant passages found for this trope."
            )
            self.trope_evidence[book_id][trope_name] = evidence
            return evidence
        
        # Format relevant passages
        relevant_passages = self._docs_to_string(relevant_docs)
        
        # Generate evidence analysis
        evidence_response = rate_limited_call(
            "google_genai",
            self.evidence_generator,
            prompt_kwargs={
                "book_id": book_id,
                "trope_name": trope_name,
                "trope_description": trope_description,
                "trope_relevance": trope_relevance,
                "relevant_passages": relevant_passages
            }
        )
        
        # Extract paragraphs from citation IDs
        paragraphs = []
        for doc in relevant_docs:
            if doc.id in evidence_response.data.citation_ids:
                paragraphs.append(doc.text)
        
        evidence_response.data.paragraphs = paragraphs
        
        # Store evidence
        self.trope_evidence[book_id][trope_name] = evidence_response.data
        
        return evidence_response.data
    
    def analyze_book_tropes(
        self,
        book_id: str,
        book_chunks: List[Dict],
        tropes: List[Dict[str, str]]
    ) -> Dict[str, TropeEvidence]:
        """
        Analyze a book for the presence of multiple tropes.
        
        Args:
            book_id: Identifier of the book
            book_chunks: List of book chunks
            tropes: List of dictionaries, each containing 'name' and 'description' keys
            
        Returns:
            Dictionary mapping trope names to their evidence in the book
        """
        logger.info(f"Analyzing book '{book_id}' for {len(tropes)} tropes")
        
        # Ensure tropes are analyzed
        self.analyze_tropes(tropes)
        
        # Find evidence for each trope
        book_trope_evidence = {}
        for trope in tropes:
            try:
                evidence = self.find_trope_evidence(
                    book_id=book_id,
                    trope_name=trope['name'],
                    trope_description=trope['description'],
                    book_chunks=book_chunks
                )
                book_trope_evidence[trope['name']] = evidence
                logger.info(f"Found evidence for trope '{trope['name']}' in book '{book_id}' - Confidence: {evidence.confidence_score}/10")
            except Exception as e:
                logger.error(f"Error finding evidence for trope '{trope['name']}' in book '{book_id}': {str(e)}")
        
        return book_trope_evidence
    
    def compare_stories(
        self,
        story1_id: str,
        story2_id: str,
        story1_trope_evidence: Dict[str, TropeEvidence],
        story2_trope_evidence: Dict[str, TropeEvidence]
    ) -> StorySimilarityResult:
        """
        Compare two stories based on their shared tropes.
        
        Args:
            story1_id: Identifier of the first story
            story2_id: Identifier of the second story
            story1_trope_evidence: Dictionary mapping trope names to their evidence in the first story
            story2_trope_evidence: Dictionary mapping trope names to their evidence in the second story
            
        Returns:
            StorySimilarityResult object containing similarity assessment
        """
        logger.info(f"Comparing stories '{story1_id}' and '{story2_id}'")
        
        # Identify shared and unique tropes
        story1_tropes = set(story1_trope_evidence.keys())
        story2_tropes = set(story2_trope_evidence.keys())
        
        shared_tropes = story1_tropes.intersection(story2_tropes)
        unique_tropes_story1 = story1_tropes - story2_tropes
        unique_tropes_story2 = story2_tropes - story1_tropes
        
        logger.info(f"Found {len(shared_tropes)} shared tropes, {len(unique_tropes_story1)} unique to '{story1_id}', {len(unique_tropes_story2)} unique to '{story2_id}'")
        
        # Format trope information for each story
        story1_tropes_info = []
        for trope_name in story1_tropes:
            evidence = story1_trope_evidence[trope_name]
            trope_analysis = self.analyzed_tropes.get(trope_name)
            relevance_score = trope_analysis.similarity_relevance_score if trope_analysis else 0
            
            story1_tropes_info.append({
                "name": trope_name,
                "confidence": evidence.confidence_score,
                "relevance": relevance_score,
                "category": trope_analysis.primary_category if trope_analysis else "unknown"
            })
        
        story2_tropes_info = []
        for trope_name in story2_tropes:
            evidence = story2_trope_evidence[trope_name]
            trope_analysis = self.analyzed_tropes.get(trope_name)
            relevance_score = trope_analysis.similarity_relevance_score if trope_analysis else 0
            
            story2_tropes_info.append({
                "name": trope_name,
                "confidence": evidence.confidence_score,
                "relevance": relevance_score,
                "category": trope_analysis.primary_category if trope_analysis else "unknown"
            })
        
        # Format shared trope evidence
        shared_trope_evidence = []
        for trope_name in shared_tropes:
            evidence1 = story1_trope_evidence[trope_name]
            evidence2 = story2_trope_evidence[trope_name]
            trope_analysis = self.analyzed_tropes.get(trope_name)
            
            shared_trope_evidence.append({
                "trope_name": trope_name,
                "story1_evidence": {
                    "confidence": evidence1.confidence_score,
                    "reasoning": evidence1.reasoning,
                    "example_paragraph": evidence1.paragraphs[0] if evidence1.paragraphs else ""
                },
                "story2_evidence": {
                    "confidence": evidence2.confidence_score,
                    "reasoning": evidence2.reasoning,
                    "example_paragraph": evidence2.paragraphs[0] if evidence2.paragraphs else ""
                },
                "relevance_score": trope_analysis.similarity_relevance_score if trope_analysis else 0,
                "category": trope_analysis.primary_category if trope_analysis else "unknown"
            })
        
        # Generate similarity analysis
        similarity_response = rate_limited_call(
            "google_genai",
            self.similarity_generator,
            prompt_kwargs={
                "story1_id": story1_id,
                "story2_id": story2_id,
                "story1_tropes": json.dumps(story1_tropes_info, default=str),
                "story2_tropes": json.dumps(story2_tropes_info, default=str),
                "shared_trope_evidence": json.dumps(shared_trope_evidence, default=str)
            }
        )
        
        return similarity_response.data
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save analysis results to a JSON file.
        
        Args:
            results: Dictionary of results
            output_file: Path to save the results
        """
        try:
            # Save to JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")


def load_book_chunks(book_path: str) -> List[Dict]:
    """
    Load book chunks from a JSONL file.
    
    Args:
        book_path: Path to the JSONL file
        
    Returns:
        List of book chunks
    """
    return load_jsonl(book_path)


def main():
    # Set up argument parser
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze tropes in books and compare story similarity')
    parser.add_argument('--tropes', type=str, required=True, help='Path to JSON file containing tropes')
    parser.add_argument('--books', type=str, nargs='+', required=True, help='Paths to JSONL files containing book chunks')
    parser.add_argument('--output', type=str, default='story_similarity_results.json', help='Path to save analysis results')
    parser.add_argument('--model', type=str, default='gemini-2.0-flash', help='Model to use for analysis')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for model generation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load tropes
    with open(args.tropes, 'r', encoding='utf-8') as f:
        tropes = json.load(f)
    logger.info(f"Loaded {len(tropes)} tropes from {args.tropes}")
    
    # Initialize model client
    model_client = adal.GoogleGenAIClient()
    model_kwargs = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    analyzer = TropeSimilarityAnalyzer(
        model_client=model_client,
        model_kwargs=model_kwargs
    )
    
    analyzed_tropes = analyzer.analyze_tropes(tropes)
    logger.info(f"Analyzed {len(analyzed_tropes)} tropes")
    
    book_analyses = {}
    for book_path in args.books:
        book_id = os.path.basename(book_path).split('.')[0]
        book_chunks = load_book_chunks(book_path)
        logger.info(f"Loaded {len(book_chunks)} chunks from book '{book_id}'")
        
        book_trope_evidence = analyzer.analyze_book_tropes(book_id, book_chunks, tropes)
        book_analyses[book_id] = book_trope_evidence
        logger.info(f"Analyzed book '{book_id}' for {len(tropes)} tropes")
    
    story_comparisons = []
    if len(args.books) > 1:
        book_ids = list(book_analyses.keys())
        for i in range(len(book_ids)):
            for j in range(i+1, len(book_ids)):
                story1_id = book_ids[i]
                story2_id = book_ids[j]
                
                similarity_result = analyzer.compare_stories(
                    story1_id=story1_id,
                    story2_id=story2_id,
                    story1_trope_evidence=book_analyses[story1_id],
                    story2_trope_evidence=book_analyses[story2_id]
                )
                
                story_comparisons.append(similarity_result.__dict__)
                logger.info(f"Compared stories '{story1_id}' and '{story2_id}' - Similarity score: {similarity_result.similarity_score}")
    
    # Save results
    results = {
        "analyzed_tropes": {name: trope.__dict__ for name, trope in analyzed_tropes.items()},
        "book_analyses": {book_id: {trope: evidence.__dict__ for trope, evidence in analysis.items()} for book_id, analysis in book_analyses.items()},
        "story_comparisons": story_comparisons
    }
    
    analyzer.save_results(results, args.output)
    
    # Print summary
    print("\n=== STORY SIMILARITY ANALYSIS SUMMARY ===\n")
    print(f"Analyzed {len(analyzed_tropes)} tropes across {len(book_analyses)} books")
    
    if story_comparisons:
        print("\nStory Comparisons:")
        for comparison in story_comparisons:
            print(f"  {comparison['story1_id']} vs {comparison['story2_id']}: Similarity Score = {comparison['similarity_score']}")
            print(f"    Shared Tropes: {len(comparison['shared_tropes'])}")
            print(f"    Unique to {comparison['story1_id']}: {len(comparison['unique_tropes_story1'])}")
            print(f"    Unique to {comparison['story2_id']}: {len(comparison['unique_tropes_story2'])}")
    
    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main() 