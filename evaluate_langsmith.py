
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_ollama import ChatOllama

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "attention.pdf"

DATASET_NAME = "attention-rag-evaluation-v1"

LANGSMITH_PROJECT = "local-agentic-rag"

JUDGE_MODEL = "llama3.2:latest"

OLLAMA_BASE_URL = "http://localhost:11434"

if not os.getenv("LANGSMITH_API_KEY"):
    raise ValueError(
        "LANGSMITH_API_KEY is missing from your .env file."
    )


# ============================================================
# 2. INITIALIZE LANGSMITH
# ============================================================

client = Client()

print("LangSmith client initialized.")
print(f"Dataset: {DATASET_NAME}")


# ============================================================
# 3. INITIALIZE LOCAL EVALUATOR MODEL
# ============================================================

judge_llm = ChatOllama(
    model=JUDGE_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

print(f"Evaluation model: {JUDGE_MODEL}")


# ============================================================
# 4. INITIALIZE RAG APPLICATION
# ============================================================

def initialize_rag():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print("\nInitializing RAG application...")

    processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )

    documents = processor.process_sources(
        [str(PDF_PATH)]
    )

    print(f"Document chunks: {len(documents)}")

    vector_store = VectorStore()
    vector_store.create_vectorstore(documents)

    retriever = vector_store.get_retriever()

    llm = Config.get_llm()

    graph_builder = GraphBuilder(
        retriever=retriever,
        llm=llm,
    )

    graph_builder.build()

    print("RAG application ready.")

    return graph_builder


graph_builder = initialize_rag()


# ============================================================
# 5. TARGET FUNCTION
# ============================================================

def rag_target(inputs: dict) -> dict:
    """
    Runs one dataset question through the complete
    LangGraph RAG application.
    """

    question = inputs["question"]

    print(f"\nRunning question: {question}")

    state = graph_builder.run(question)

    if isinstance(state, dict):
        answer = state.get("answer", "")
        relevant_docs = state.get("relevant_docs", [])
        is_relevant = state.get("is_relevant", False)

    else:
        answer = getattr(state, "answer", "")
        relevant_docs = getattr(state, "relevant_docs", [])
        is_relevant = getattr(state, "is_relevant", False)

    contexts = [
        doc.page_content
        for doc in relevant_docs
        if hasattr(doc, "page_content")
    ]

    citations = re.findall(
        r"\[([^\]]+)\]",
        answer,
    )

    print(f"Retrieved relevant contexts: {len(contexts)}")
    print(f"Answer: {answer}")

    return {
        "answer": str(answer),
        "contexts": contexts,
        "citations": citations,
        "is_relevant": bool(is_relevant),
    }


# ============================================================
# 6. LLM-AS-A-JUDGE HELPER
# ============================================================

def llm_judge(
    criterion: str,
    question: str,
    answer: str,
    reference: str = "",
    contexts: list = None,
) -> dict:
    """
    Ask the local Ollama model to score one criterion.

    Expected response:
        SCORE: 4
        REASON: Explanation
    """

    contexts = contexts or []

    context_text = "\n\n".join(
        f"Context {i + 1}: {context}"
        for i, context in enumerate(contexts)
    )

    prompt = f"""
You are an evaluator for a Retrieval Augmented Generation system.

Evaluate the response using the following criterion:

{criterion}

Question:
{question}

Generated answer:
{answer}

Reference answer:
{reference if reference else "Not provided"}

Retrieved contexts:
{context_text if context_text else "No contexts retrieved"}

Scoring scale:
5 = Excellent
4 = Good
3 = Partially correct
2 = Mostly incorrect
1 = Very poor
0 = Completely incorrect or unsupported

Rules:
- Evaluate only the requested criterion.
- Do not assume information that is not provided.
- For faithfulness, check whether the answer is supported by
  the retrieved contexts.
- For correctness, compare the answer with the reference.
- For retrieval relevance, assess whether the contexts contain
  useful information for answering the question.
- Return a score from 0 to 5.

Return exactly this format:

SCORE: 4
REASON: Your explanation here
"""

    response = judge_llm.invoke(prompt)

    content = (
        response.content
        if hasattr(response, "content")
        else str(response)
    )

    score_match = re.search(
        r"SCORE\s*:\s*([0-5](?:\.\d+)?)",
        content,
        re.IGNORECASE,
    )

    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        content,
        re.IGNORECASE | re.DOTALL,
    )

    if not score_match:
        raise ValueError(
            f"Evaluator returned an invalid score: {content}"
        )

    score = float(score_match.group(1))

    reason = (
        reason_match.group(1).strip()
        if reason_match
        else content.strip()
    )

    return {
        "score": score / 5.0,
        "reason": reason,
    }


# ============================================================
# 7. CORRECTNESS EVALUATOR
# ============================================================

def correctness_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
):
    """
    Measures answer correctness against the reference answer.
    """

    result = llm_judge(
        criterion=(
            "Answer correctness: determine whether the generated "
            "answer correctly addresses the question and conveys "
            "the important information in the reference answer."
        ),
        question=inputs["question"],
        answer=outputs["answer"],
        reference=reference_outputs["reference"],
        contexts=outputs["contexts"],
    )

    return {
        "key": "correctness",
        "score": result["score"],
        "comment": result["reason"],
    }


# ============================================================
# 8. FAITHFULNESS EVALUATOR
# ============================================================

def faithfulness_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
):
    """
    Measures whether the generated answer is supported
    by the retrieved contexts.
    """

    if not outputs["contexts"]:
        return {
            "key": "faithfulness",
            "score": 0.0,
            "comment": "No relevant contexts were retrieved.",
        }

    result = llm_judge(
        criterion=(
            "Faithfulness: determine whether the factual claims "
            "in the generated answer are supported by the "
            "retrieved contexts. Penalize unsupported claims."
        ),
        question=inputs["question"],
        answer=outputs["answer"],
        contexts=outputs["contexts"],
    )

    return {
        "key": "faithfulness",
        "score": result["score"],
        "comment": result["reason"],
    }


# ============================================================
# 9. RETRIEVAL RELEVANCE EVALUATOR
# ============================================================

def retrieval_relevance_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
):
    """
    Measures the relevance of retrieved passages.
    """

    if not outputs["contexts"]:
        return {
            "key": "retrieval_relevance",
            "score": 0.0,
            "comment": "No relevant contexts were retrieved.",
        }

    result = llm_judge(
        criterion=(
            "Retrieval relevance: evaluate whether the retrieved "
            "passages contain information that is useful and "
            "relevant to answering the question. Consider the "
            "retrieved passages collectively."
        ),
        question=inputs["question"],
        answer=outputs["answer"],
        reference=reference_outputs["reference"],
        contexts=outputs["contexts"],
    )

    return {
        "key": "retrieval_relevance",
        "score": result["score"],
        "comment": result["reason"],
    }


# ============================================================
# 10. CITATION EVALUATOR
# ============================================================

def citation_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
):
    """
    Checks citation formatting and sentence coverage.

    This is a formatting audit, not proof that a citation
    actually supports the associated claim.
    """

    answer = outputs["answer"]

    citations = re.findall(
        r"\[([^\]]+)\]",
        answer,
    )

    valid_citations = [
        citation
        for citation in citations
        if (
            ".pdf" in citation.lower()
            and re.search(
                r"\bp\.\s*\d+",
                citation.lower(),
            )
        )
    ]

    invalid_citations = [
        citation
        for citation in citations
        if citation not in valid_citations
    ]

    # Protect citations before sentence splitting.
    protected_answer = re.sub(
        r"\[[^\]]+\]",
        " CITATIONMARKER ",
        answer,
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        protected_answer.strip(),
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    abstention_phrases = [
        "provided documents do not contain",
        "provided document passages do not",
        "insufficient information to answer",
    ]

    is_abstention = any(
        phrase in answer.lower()
        for phrase in abstention_phrases
    )

    if is_abstention:
        sentences = []

    cited_sentences = [
        sentence
        for sentence in sentences
        if "CITATIONMARKER" in sentence
    ]

    coverage = (
        len(cited_sentences) / len(sentences)
        if sentences
        else 0.0
    )

    precision = (
        len(valid_citations) / len(citations)
        if citations
        else 0.0
    )

    # Combined formatting and coverage score.
    score = (
        (precision + coverage) / 2
    )

    return {
        "key": "citation_accuracy",
        "score": score,
        "comment": (
            f"Valid citations: {len(valid_citations)}; "
            f"Invalid citations: {len(invalid_citations)}; "
            f"Citation precision: {precision:.2f}; "
            f"Sentence coverage: {coverage:.2f}."
        ),
    }


# ============================================================
# 11. RUN LANGSMITH EXPERIMENT
# ============================================================

def run_evaluation():

    print("\n" + "=" * 65)
    print("STARTING LANGSMITH EVALUATION")
    print("=" * 65)

    evaluators = [
        correctness_evaluator,
        faithfulness_evaluator,
        retrieval_relevance_evaluator,
        citation_evaluator,
    ]

    results = evaluate(
        rag_target,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix="local-agentic-rag-evaluation",
        description=(
            "Evaluation of local Agentic RAG using LangGraph, "
            "Ollama, FAISS and BGE embeddings."
        ),
        metadata={
            "rag_model": Config.LLM_MODEL,
            "judge_model": JUDGE_MODEL,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "vector_database": "FAISS",
        },
        max_concurrency=1,
    )

    print("\n" + "=" * 65)
    print("LANGSMITH EVALUATION COMPLETED")
    print("=" * 65)

    print("Experiment name:", results.experiment_name)

    print(
        "\nOpen your LangSmith project to inspect "
        "the experiment results."
    )


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":
    run_evaluation()
