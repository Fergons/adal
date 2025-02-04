
# Narrative Trope RAG Pipeline

A synthetic data generation system that uses RAG (Retrieval Augmented Generation) to analyze narrative tropes from TVTropes in books and stories.

## Overview

This project implements an iterative RAG pipeline that:

1. Takes a narrative trope query and book text as input
2. Uses FAISS embedding-based retrieval to find relevant text passages
3. Generates clarifying questions to refine the search when needed
4. Produces query-focused summaries of the book content
5. Outputs structured determinations of whether the trope exists in the text

## Key Components

- FAISS vector store for efficient similarity search
- Query refinement through clarifying questions
- Chunk-wise summarization focused on trope relevance 
- Structured output with citations to source text
- Tracing and logging of the generation process

## Usage

The system can be used to:
- Generate synthetic training data by analyzing tropes in books
- Build datasets mapping narrative patterns to text evidence
- Study how tropes manifest across different works

## Requirements

- Python 3.10+
- FAISS for vector similarity search
- An LLM service for text generation (e.g. Ollama)
- Book text data in chunked format

