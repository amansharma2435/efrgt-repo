#!/usr/bin/env python3
"""Generate six-week Amity-style internship progress report DOCX."""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

LOGO_PATH = Path("/tmp/amity_logo.jpg")
OUTPUT_PATH = Path("/workspace/AI_RAG_Six_Weeks_Progress_Report.docx")

PROJECT_TITLE = (
    "AI-Powered Document Question Answering System using "
    "Retrieval-Augmented Generation (RAG)"
)

WEEKS = [
    {
        "number": "1",
        "progress": (
            "During this week, the internship project was formally introduced and the "
            "assigned project description document was reviewed in detail. The core "
            "problem statement was understood: users need to upload organisational "
            "documents (PDFs, policies, manuals, reports) and obtain accurate answers "
            "to natural-language questions grounded in that content, rather than "
            "relying on generic model knowledge alone.\n\n"
            "The primary objectives were identified as designing a retrieval-augmented "
            "question-answering workflow, enabling document ingestion, semantic search "
            "over indexed content, and generation of context-aware responses with "
            "appropriate source attribution. The project scope was clarified to cover "
            "single-document and multi-document RAG fundamentals, with optional API or "
            "UI layers noted as later enhancements.\n\n"
            "Fundamental concepts of Retrieval-Augmented Generation were studied, "
            "including the limitation of standalone large language models (hallucination "
            "and lack of private data access), and how external retrieval supplies "
            "evidence to the generator. The end-to-end pipeline—upload, text extraction, "
            "chunking, embedding, vector storage, retrieval, prompt construction, and "
            "LLM response—was mapped at a conceptual level. No implementation was "
            "undertaken; the focus remained on comprehension, terminology, and alignment "
            "with internship learning outcomes."
        ),
    },
    {
        "number": "2",
        "progress": (
            "This week focused on structured requirement analysis and architectural "
            "planning based on the project brief. Functional requirements were "
            "documented: secure document upload, text extraction from PDFs, indexing "
            "of processed content, natural-language querying, ranked retrieval of "
            "relevant passages, and generation of answers constrained to retrieved "
            "context. Non-functional requirements such as response relevance, "
            "traceability to source pages, modularity, and extensibility were also "
            "noted for future design decisions.\n\n"
            "The proposed system architecture was studied following the prescribed "
            "flow: PDF → text extraction → chunking → embeddings → vector database → "
            "semantic retrieval → LLM → final answer. Candidate API endpoints "
            "(/documents/upload, /documents/process, /ask) were reviewed as a reference "
            "design for later backend development. The roles of the embedding model, "
            "vector index, and orchestration layer (e.g., LangChain) within a RAG "
            "application were examined.\n\n"
            "A comparative survey of the suggested technology stack was initiated: "
            "Python as the core language; PyPDF2/PyMuPDF for parsing; OpenAI or Azure "
            "OpenAI for embeddings and chat completion; FAISS or ChromaDB for vector "
            "storage; and optional FastAPI or Streamlit for interaction. A phased "
            "development workflow was drafted (environment setup, ingestion module, "
            "indexing module, retrieval module, generation module, guardrails, and "
            "testing). All activities remained at the analysis and planning stage."
        ),
    },
    {
        "number": "3",
        "progress": (
            "Preparation of the Python development environment was planned and "
            "documented for subsequent setup on the development machine. The intended "
            "structure includes a dedicated virtual environment, dependency management "
            "via requirements.txt, and separation of modules for ingestion, indexing, "
            "retrieval, and response generation. Version-control practices using Git "
            "were reviewed to support incremental development once coding begins.\n\n"
            "PDF processing libraries named in the project specification—PyPDF2 and "
            "PyMuPDF—were explored through official documentation and illustrative "
            "examples. Their capabilities for page-wise text extraction, handling of "
            "multi-page documents, and known limitations (scanned PDFs, layout "
            "complexity, encoding issues) were compared. Sample extraction logic from "
            "the project guide was analysed to understand how raw text would be "
            "assembled before cleaning and chunking.\n\n"
            "The document upload and text-extraction stage of the workflow was "
            "specified in a preliminary design: accept uploaded files, validate format "
            "and size, persist or buffer the file, invoke the chosen parser, and "
            "normalise extracted text (whitespace, line breaks, optional metadata such "
            "as page numbers). Integration points with a future /documents/upload and "
            "/documents/process API were outlined. Hands-on coding and live testing "
            "were not performed during this reporting period; the week emphasised "
            "tooling research and module-level design."
        ),
    },
    {
        "number": "4",
        "progress": (
            "Attention was directed toward the indexing pipeline components that "
            "follow text extraction in the RAG architecture. Text chunking strategies "
            "were studied, including fixed-size splits with overlap, sentence-aware "
            "segmentation, and the trade-off between chunk granularity and retrieval "
            "precision. Criteria for chunk size, overlap percentage, and preservation "
            "of metadata (source file name, page number) were documented for later "
            "implementation.\n\n"
            "Embedding generation was reviewed conceptually: converting each text chunk "
            "into a dense vector representation that captures semantic meaning, using "
            "models such as those provided through OpenAI or Azure OpenAI APIs. "
            "Consistency requirements were noted—the same embedding model must be used "
            "for both document chunks and user queries to ensure comparable vector "
            "space. Batch processing and rate-limit considerations were listed as "
            "operational factors for the processing stage.\n\n"
            "Vector database integration options (FAISS for local similarity search "
            "and ChromaDB for persistent collection management) were compared with "
            "respect to ease of setup, persistence, and scalability for an internship-scale "
            "prototype. A planned indexing workflow was defined: chunk extracted text, "
            "generate embeddings, assign unique identifiers, store vectors with "
            "metadata, and expose a process step to rebuild or update the index when "
            "new documents are added. Development of these modules is scheduled for a "
            "subsequent phase; this week consolidated design notes and technical "
            "reading only."
        ),
    },
    {
        "number": "5",
        "progress": (
            "The retrieval and generation stages of the RAG pipeline were analysed in "
            "depth during this week. The semantic retrieval mechanism was studied: "
            "embed the user question, perform similarity search (e.g., cosine similarity "
            "or inner product) against the vector store, and retrieve the top-k chunks "
            "most relevant to the query. Parameters such as k, score thresholds, and "
            "optional reranking were identified as tunable design choices affecting "
            "answer quality.\n\n"
            "Prompt engineering principles applicable to RAG were reviewed, including "
            "clear system instructions, injection of retrieved context, explicit "
            "instruction to answer only from provided passages, and formatting that "
            "separates context from the user question. Example prompt templates aligned "
            "with the project’s /ask use case were drafted on paper for later coding.\n\n"
            "Integration of a large language model into the workflow was mapped: after "
            "retrieval, assemble context and question, invoke the chat completion API, "
            "and return the model response together with source references (file name "
            "and page). The orchestration role of frameworks such as LangChain was "
            "explored through documentation and architecture diagrams. End-to-end "
            "sequence from user question to grounded answer was validated logically "
            "against the internship project description. Implementation of retrieval "
            "and LLM calls remains planned; no production API or live model integration "
            "was completed in this week."
        ),
    },
    {
        "number": "6",
        "progress": (
            "The final planning week consolidated quality, safety, and completion "
            "activities for the forthcoming implementation phase. Guardrail strategies "
            "were studied, particularly instructing the system to decline answering when "
            "retrieved context is insufficient or irrelevant, thereby reducing "
            "hallucinated responses. Threshold-based retrieval checks and explicit "
            "fallback messages were documented as recommended behaviours.\n\n"
            "A structured testing and error-handling plan was prepared covering unit "
            "tests for text extraction and chunking, integration tests for the indexing "
            "and query path, and manual test cases with sample policy-style questions "
            "and expected source citations. Error scenarios—corrupt PDFs, empty "
            "extraction, API failures, and missing index entries—were enumerated with "
            "proposed handling responses.\n\n"
            "The complete end-to-end workflow was reviewed against the project "
            "specification from document upload through grounded answer generation. "
            "Gaps between the current planning status and full deployment (optional "
            "FastAPI endpoints, Streamlit UI, Docker, cloud hosting) were recorded for "
            "future sprints. Preparation for project documentation was initiated, "
            "including an outline for user guide, architecture diagram, setup "
            "instructions, and weekly progress continuity. Comprehensive implementation, "
            "formal test execution, and deployment are intended as the next project "
            "phase beyond this six-week preparatory reporting cycle."
        ),
    },
]


def set_a4_margins(section):
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)


def add_horizontal_rule(paragraph):
    p = paragraph._p
    p_pr = p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "auto")
    p_bdr.append(bottom)
    p_pr.append(p_bdr)


def styled_run(paragraph, text, bold=False, size=11, underline=False):
    run = paragraph.add_run(text)
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    run.font.size = Pt(size)
    run.bold = bold
    run.underline = underline
    return run


def add_report_page(doc, week_data, add_break_after=True):
    # Logo
    if LOGO_PATH.exists():
        logo_para = doc.add_paragraph()
        logo_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = logo_para.add_run()
        run.add_picture(str(LOGO_PATH), width=Inches(2.8))

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    styled_run(title, "Weekly Progress Report", bold=True, size=18)

    subject = doc.add_paragraph()
    subject.alignment = WD_ALIGN_PARAGRAPH.CENTER
    styled_run(subject, "Subject: Internship/ Major Project/ Seminar", bold=True, size=14)

    doc.add_paragraph()

    p = doc.add_paragraph()
    styled_run(p, "Course Code:", bold=True)
    styled_run(p, " _______________________ ")

    p = doc.add_paragraph()
    styled_run(p, "Title:", bold=True)
    styled_run(p, f" {PROJECT_TITLE}")

    p = doc.add_paragraph()
    styled_run(p, "Week Number: ", bold=True)
    styled_run(p, week_data["number"])

    p = doc.add_paragraph()
    styled_run(p, "Start Date:        ", bold=True)
    styled_run(p, "DD/MM/YYYY")

    p = doc.add_paragraph()
    styled_run(p, "End Date:          ", bold=True)
    styled_run(p, "DD/MM/YYYY")

    doc.add_paragraph()

    heading = doc.add_paragraph()
    styled_run(heading, "Progress of the Project:", bold=True)

    body = doc.add_paragraph()
    body.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    body.paragraph_format.space_after = Pt(6)
    styled_run(body, week_data["progress"], size=11)

    doc.add_paragraph()

    sig = doc.add_paragraph()
    styled_run(sig, "Signature:", bold=True)
    styled_run(sig, "\t\t\t\t\t\t\t\tSignature:")

    doc.add_paragraph()
    doc.add_paragraph()

    row1 = doc.add_paragraph()
    styled_run(row1, "Name of the Student: ", bold=False)
    styled_run(row1, "___________________________")
    styled_run(row1, "\t\tName of Supervisor: ", bold=False)
    styled_run(row1, "___________________________")

    row2 = doc.add_paragraph()
    styled_run(row2, "Enrollment No.: ", bold=False)
    styled_run(row2, "___________________________")
    styled_run(row2, "\t\tEmployee ID: ", bold=False)
    styled_run(row2, "___________________________")

    row3 = doc.add_paragraph()
    styled_run(row3, "Course: ", bold=False)
    styled_run(row3, "___________________________")

    row4 = doc.add_paragraph()
    styled_run(row4, "Semester: ", bold=False)
    styled_run(row4, "___________________________")

    if add_break_after:
        pb = doc.add_paragraph()
        run = pb.add_run()
        run.add_break(WD_BREAK.PAGE)


def main():
    doc = Document()
    set_a4_margins(doc.sections[0])

    for idx, week in enumerate(WEEKS):
        add_report_page(doc, week, add_break_after=idx < len(WEEKS) - 1)

    doc.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
