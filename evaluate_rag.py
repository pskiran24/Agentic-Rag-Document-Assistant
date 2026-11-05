
import asyncio
import re
from pathlib import Path

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
)

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "attention.pdf"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2:latest"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 4


# ============================================================
# 2. EVALUATION DATASET
# ============================================================

TEST_CASES = [
    {
        "question": "What is self-attention?",
        "reference": (
            "Self-attention is an attention mechanism that relates "
            "different positions of a single sequence when computing "
            "a representation of that sequence."
        ),
    },
    {
        "question": "What is the purpose of multi-head attention?",
        "reference": (
            "Multi-head attention allows the model to jointly attend "
            "to information from different representation subspaces "
            "at different positions."
        ),
    },
    {
        "question": "How does scaled dot-product attention work?",
        "reference": (
            "Scaled dot-product attention computes the dot products "
            "of queries and keys, divides the results by the square "
            "root of the key dimension, applies softmax, and uses "
            "the resulting weights to combine the values."
        ),
    },
    {
        "question": "How many attention heads are used in the original Transformer?",
        "reference": (
            "The original Transformer uses eight attention heads "
            "in its multi-head attention layers."
        ),
    },
    {
        "question": "Why does scaled dot-product attention divide by the square root of dk?",
        "reference": (
            "When the key dimension is large, dot products can become "
            "large in magnitude. This can push the softmax function "
            "into regions with extremely small gradients. Dividing "
            "by the square root of dk helps counteract this effect."
        ),
    },
]


# ============================================================
# 3. INITIALIZE RAGAS EVALUATOR
# ============================================================

def initialize_evaluator():

    ollama_client = AsyncOpenAI(
        api_key="ollama",
        base_url=OLLAMA_BASE_URL,
        timeout=180.0,
    )

    # Use Ollama through its OpenAI-compatible endpoint.
    # Do not pass instructor.Mode.MD_JSON here because it
    # caused a duplicate mode argument in the previous run.

    evaluator_llm = llm_factory(
        model=OLLAMA_MODEL,
        provider="openai",
        client=ollama_client,
        temperature=0,
    )

    evaluator_embeddings = embedding_factory(
        provider="huggingface",
        model=EMBEDDING_MODEL,
        device="cpu",
    )

    return evaluator_llm, evaluator_embeddings


# ============================================================
# 4. INITIALIZE RAG APPLICATION
# ============================================================

def initialize_rag():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH}"
        )

    print("\nInitializing RAG application...")

    processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )

    documents = processor.process_sources(
        [str(PDF_PATH)]
    )

    print(f"Document chunks created: {len(documents)}")

    vector_store = VectorStore()
    vector_store.create_vectorstore(documents)

    retriever = vector_store.get_retriever()

    llm = Config.get_llm()

    graph_builder = GraphBuilder(
        retriever=retriever,
        llm=llm,
    )

    graph_builder.build()

    print("RAG application initialized successfully.")

    return graph_builder


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def get_state_value(state, key, default=None):

    if isinstance(state, dict):
        return state.get(key, default)

    return getattr(state, key, default)


def extract_contexts(state):

    documents = get_state_value(
        state,
        "relevant_docs",
        [],
    )

    contexts = []

    for document in documents:

        if hasattr(document, "page_content"):
            contexts.append(document.page_content)

    return contexts


def extract_answer(state):

    answer = get_state_value(
        state,
        "answer",
        "",
    )

    return str(answer).strip()


def get_metric_value(result):

    if hasattr(result, "value"):
        return result.value

    return result


# ============================================================
# 6. CITATION AUDIT
# ============================================================

def audit_citations(answer: str) -> dict:

    citation_pattern = r"\[([^\]]+)\]"

    citations = re.findall(
        citation_pattern,
        answer,
    )

    # Identify standard abstention responses.
    abstention_phrases = [
        "provided documents do not contain",
        "provided document passages do not",
        "insufficient information to answer",
    ]

    is_abstention = any(
        phrase in answer.lower()
        for phrase in abstention_phrases
    )

    # Validate citation formatting.
    valid_citations = []
    invalid_citations = []

    for citation in citations:

        citation_lower = citation.lower()

        if (
            ".pdf" in citation_lower
            and re.search(r"\bp\.\s*\d+", citation_lower)
        ):
            valid_citations.append(citation)

        else:
            invalid_citations.append(citation)

    # Replace citations before splitting sentences.
    # This prevents periods in filenames and page labels
    # from being interpreted as sentence boundaries.

    protected_answer = re.sub(
        citation_pattern,
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

    if is_abstention:
        sentences = []

    cited_sentences = [
        sentence
        for sentence in sentences
        if "CITATIONMARKER" in sentence
    ]

    total_sentences = len(sentences)

    coverage = (
        len(cited_sentences) / total_sentences
        if total_sentences > 0
        else 0.0
    )

    citation_precision = (
        len(valid_citations) / len(citations)
        if citations
        else 0.0
    )

    return {
        "total_citations": len(citations),
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "citation_precision": citation_precision,
        "total_sentences": total_sentences,
        "cited_sentences": len(cited_sentences),
        "sentence_citation_coverage": coverage,
    }


# ============================================================
# 7. RUN RAG TEST CASES
# ============================================================

def run_rag_tests(graph_builder):

    results = []

    print("\n" + "=" * 65)
    print("RUNNING RAG TEST CASES")
    print("=" * 65)

    for index, test_case in enumerate(TEST_CASES, start=1):

        question = test_case["question"]

        print(f"\nTest case {index}: {question}")

        try:

            state = graph_builder.run(question)

            answer = extract_answer(state)
            contexts = extract_contexts(state)

            results.append({
                "question": question,
                "reference": test_case["reference"],
                "answer": answer,
                "contexts": contexts,
            })

            print(f"Retrieved contexts: {len(contexts)}")
            print(f"Answer: {answer}")

        except Exception as error:

            print(f"RAG execution failed: {error}")

            results.append({
                "question": question,
                "reference": test_case["reference"],
                "answer": "",
                "contexts": [],
            })

    return results


# ============================================================
# 8. RAGAS METRIC SCORING
# ============================================================

async def score_metric(
    metric,
    metric_name,
    question,
    answer,
    reference,
    contexts,
):

    if not answer or not contexts:

        print(
            f"  {metric_name}: N/A "
            "(answer or retrieved contexts missing)"
        )

        return None

    try:

        if metric_name == "Faithfulness":

            result = await metric.ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )

        elif metric_name == "Answer Relevancy":

            result = await metric.ascore(
                user_input=question,
                response=answer,
            )

        elif metric_name == "Context Precision":

            result = await metric.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=contexts,
            )

        elif metric_name == "Context Recall":

            result = await metric.ascore(
                user_input=question,
                retrieved_contexts=contexts,
                reference=reference,
            )

        else:

            raise ValueError(
                f"Unknown metric: {metric_name}"
            )

        score = get_metric_value(result)

        print(f"  {metric_name}: {score}")

        return score

    except Exception as error:

        print(
            f"  {metric_name}: N/A "
            f"(evaluation failed: {type(error).__name__}: {error})"
        )

        return None


# ============================================================
# 9. EVALUATE ALL RESULTS
# ============================================================

async def evaluate_results(results):

    evaluator_llm, evaluator_embeddings = (
        initialize_evaluator()
    )

    metrics = {

        "Faithfulness": Faithfulness(
            llm=evaluator_llm,
        ),

        "Answer Relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),

        "Context Precision": ContextPrecisionWithReference(
            llm=evaluator_llm,
        ),

        "Context Recall": ContextRecall(
            llm=evaluator_llm,
        ),
    }

    metric_scores = {
        name: []
        for name in metrics
    }

    citation_results = []

    print("\n" + "=" * 65)
    print("RAGAS EVALUATION")
    print("=" * 65)

    for index, result in enumerate(results, start=1):

        question = result["question"]
        answer = result["answer"]
        reference = result["reference"]
        contexts = result["contexts"]

        print(f"\nEvaluation {index}: {question}")

        for metric_name, metric in metrics.items():

            score = await score_metric(
                metric=metric,
                metric_name=metric_name,
                question=question,
                answer=answer,
                reference=reference,
                contexts=contexts,
            )

            if score is not None:

                try:
                    metric_scores[metric_name].append(
                        float(score)
                    )

                except (TypeError, ValueError):
                    pass

        citation_audit = audit_citations(answer)

        citation_results.append(citation_audit)

        print("\nCitation audit:")

        print(
            "  Valid citations:",
            citation_audit["valid_citations"],
        )

        print(
            "  Invalid citations:",
            citation_audit["invalid_citations"],
        )

        print(
            "  Citation precision:",
            round(
                citation_audit["citation_precision"],
                4,
            ),
        )

        print(
            "  Sentence citation coverage:",
            round(
                citation_audit["sentence_citation_coverage"],
                4,
            ),
        )

    return metric_scores, citation_results


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

def print_summary(
    metric_scores,
    citation_results,
):

    print("\n" + "=" * 65)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 65)

    print("\nRAGAS METRICS")

    for metric_name, scores in metric_scores.items():

        if scores:

            average = sum(scores) / len(scores)

            print(
                f"{metric_name:25s}: "
                f"{average:.4f} "
                f"({len(scores)} successful evaluations)"
            )

        else:

            print(
                f"{metric_name:25s}: N/A "
                "(no successful evaluations)"
            )

    print("\nCITATION METRICS")

    total_valid = sum(
        len(item["valid_citations"])
        for item in citation_results
    )

    total_invalid = sum(
        len(item["invalid_citations"])
        for item in citation_results
    )

    total_citations = total_valid + total_invalid

    overall_precision = (
        total_valid / total_citations
        if total_citations > 0
        else 0.0
    )

    total_sentences = sum(
        item["total_sentences"]
        for item in citation_results
    )

    total_cited_sentences = sum(
        item["cited_sentences"]
        for item in citation_results
    )

    overall_coverage = (
        total_cited_sentences / total_sentences
        if total_sentences > 0
        else 0.0
    )

    print(f"Valid citations: {total_valid}")
    print(f"Invalid citations: {total_invalid}")

    print(
        f"Citation precision: {overall_precision:.4f}"
    )

    print(
        f"Sentence citation coverage: {overall_coverage:.4f}"
    )

    print("\n" + "=" * 65)


# ============================================================
# 11. MAIN
# ============================================================

async def main():

    print("=" * 65)
    print("LOCAL RAG EVALUATION")
    print("=" * 65)

    print(f"LLM: {OLLAMA_MODEL}")
    print(f"Embeddings: {EMBEDDING_MODEL}")
    print(f"PDF: {PDF_PATH}")

    # Initialize RAG.
    graph_builder = initialize_rag()

    # Generate answers.
    results = run_rag_tests(graph_builder)

    # Evaluate answers.
    metric_scores, citation_results = await evaluate_results(
        results
    )

    # Print final scores.
    print_summary(
        metric_scores,
        citation_results,
    )


if __name__ == "__main__":
    asyncio.run(main())
