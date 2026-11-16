"""Document question answering with retrieval-augmented generation.

The package is split into small modules so each stage of the pipeline can be
read, tested, and explained on its own:

1. ``pdf_processing`` reads a PDF page by page.
2. ``text_chunking`` cuts those pages into overlapping passages.
3. ``embeddings`` turns passages and questions into vectors.
4. ``vector_database`` stores the vectors and finds the closest ones.
5. ``rag_pipeline`` asks Gemini to answer from those passages only.
6. ``app`` is the Streamlit interface.
"""

__version__ = "1.0.0"
