
from typing import List
import re
from pathlib import Path

from src.state.rag_state import RAGState
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


class RAGNodes:
    NO_ANSWER = (
        "The provided documents do not contain sufficient information "
        "to answer this question."
    )

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    # NODE 1: Retrieve top-k passages.
    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)

        print(f"\nRetrieved documents: {len(docs)}")

        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    # NODE 2: Grade retrieved passages.
    def grade_documents(self, state: RAGState) -> RAGState:
        relevant_docs: List[Document] = []

        grading_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a careful document relevance classifier.

Determine whether the passage contains information that directly
answers the question OR provides useful evidence for answering it.

Rules:
- Judge meaning, not exact keyword overlap.
- Direct definitions and explanations are RELEVANT.
- Supporting details, examples, formulas and parameters are RELEVANT
  when they help answer the question.
- A passage may answer only part of a question and still be RELEVANT.
- Do not mark a passage relevant merely because it discusses the
  same broad subject.
- Return exactly one token: RELEVANT or IRRELEVANT.

Example:
Question: What is the purpose of multi-head attention?
Passage: Multi-head attention allows the model to jointly attend to
information from different representation subspaces.
Decision: RELEVANT

Example:
Question: Explain quantum computing.
Passage: Self-attention relates different positions of a sequence.
Decision: IRRELEVANT
"""),
            ("human", """
Question:
{question}

Passage:
{document}

Decision:
""")
        ])

        grading_chain = grading_prompt | self.llm

        for i, doc in enumerate(state.retrieved_docs, start=1):
            result = grading_chain.invoke({
                "question": state.question,
                "document": doc.page_content
            })

            raw = str(result.content).strip().upper()

            print(f"\nPassage {i} grader response: {raw}")

            # Check IRRELEVANT first.
            token = re.search(r"\b(IRRELEVANT|RELEVANT)\b", raw)

            if token and token.group(1) == "RELEVANT":
                relevant_docs.append(doc)

        print(f"\nRelevant passages: {len(relevant_docs)}")
        print(
            f"Irrelevant passages: "
            f"{len(state.retrieved_docs) - len(relevant_docs)}"
        )

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            relevant_docs=relevant_docs,
            is_relevant=bool(relevant_docs)
        )

    # Prepare context and source mapping.
    @staticmethod
    def _build_context_and_citations(docs: List[Document]):
        context_parts = []
        citation_map = {}

        for i, doc in enumerate(docs, start=1):
            citation_id = f"D{i}"

            source = str(
                doc.metadata.get("source", "Unknown source")
            )

            filename = (
                Path(source).name
                if source != "Unknown source"
                else source
            )

            # PyPDFLoader page numbering starts from zero.
            page = doc.metadata.get("page")

            page_number = (
                int(page) + 1
                if page is not None
                else None
            )

            label = (
                f"{filename}, p. {page_number}"
                if page_number is not None
                else filename
            )

            citation_map[citation_id] = label

            context_parts.append(
                f"[{citation_id}]\n"
                f"Source: {filename}\n"
                f"PDF page: "
                f"{page_number if page_number is not None else 'not available'}\n"
                f"Passage:\n{doc.page_content}"
            )

        return "\n\n".join(context_parts), citation_map

    # Normalize citation formats and validate IDs.
    @staticmethod
    def _normalize_and_validate_citations(
        answer: str,
        citation_map: dict
    ):
        # Only recognize explicit citation formats:
        # [D1], (D1), or D1.
        pattern = (
            r"(?<![A-Za-z0-9])"
            r"(?:\[|\()?"
            r"(D\d+)"
            r"(?:\]|\))?"
            r"(?![A-Za-z0-9])"
        )

        found = re.findall(pattern, answer)

        invalid = sorted({
            cid for cid in found
            if cid not in citation_map
        })

        valid = sorted(
            {cid for cid in found if cid in citation_map},
            key=lambda x: int(x[1:])
        )

        # Replace valid IDs with readable source citations.
        def replace_citation(match):
            cid = match.group(1)

            if cid in citation_map:
                return f"[{citation_map[cid]}]"

            return match.group(0)

        normalized = re.sub(
            pattern,
            replace_citation,
            answer
        )

        # Remove duplicate adjacent source citations.
        normalized = re.sub(
            r"(\[[^\]]+\])(?:\s*\1)+",
            r"\1",
            normalized
        )

        return normalized.strip(), valid, invalid

    # NODE 3: Generate answer and validate citations.
    def generate_answer(self, state: RAGState) -> RAGState:

        if not state.relevant_docs:
            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                relevant_docs=[],
                is_relevant=False,
                answer=self.NO_ANSWER
            )

        context, citation_map = (
            self._build_context_and_citations(
                state.relevant_docs
            )
        )

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a precise document-grounded question-answering assistant.

Answer the user's question using ONLY the supplied passages.

STRICT RULES:

1. Answer the question directly and concisely.
2. Every factual sentence MUST contain at least one citation ID.
3. Use the exact citation IDs supplied in the context:
   [D1], [D2], etc.
4. Never invent citation IDs.
5. Do not use outside knowledge.
6. Do not add unsupported explanations, assumptions or implications.
7. If multiple passages support different parts of the answer,
   cite the appropriate passages.
8. If the passages only partially answer the question,
   provide the supported information and clearly state the limitation.
9. If the passages do not contain the answer, return exactly:
   The provided documents do not contain sufficient information
   to answer this question.

IMPORTANT:
- Do not output a citation list separately.
- Put citations immediately after the claims they support.
- Do not repeat the same citation unnecessarily.
- Return only the answer.

DOCUMENT PASSAGES:

{context}
"""),
            ("human", "{question}")
        ])

        answer_chain = answer_prompt | self.llm

        draft = answer_chain.invoke({
            "context": context,
            "question": state.question
        })

        answer = str(draft.content).strip()

        print("\nGenerated draft:")
        print(answer)

        # Validate citations.
        answer, valid_ids, invalid_ids = (
            self._normalize_and_validate_citations(
                answer,
                citation_map
            )
        )

        if invalid_ids:
            print(
                f"WARNING: Invalid citation IDs: {invalid_ids}"
            )

        # Retry once if the model forgot to include citations.
        if not valid_ids and self.NO_ANSWER.lower() not in answer.lower():

            print(
                "\nWARNING: No valid citations found. "
                "Retrying answer generation..."
            )

            retry_prompt = ChatPromptTemplate.from_messages([
                ("system", """
Answer the question using ONLY the supplied document passages.

You MUST include at least one valid citation ID in square brackets
for every factual claim.

Available citation IDs:
{citation_ids}

Do not invent citations or use outside knowledge.

If the passages do not answer the question, return exactly:
The provided documents do not contain sufficient information
to answer this question.

Return only the answer.

Passages:
{context}
"""),
                ("human", "{question}")
            ])

            retry_result = (retry_prompt | self.llm).invoke({
                "context": context,
                "question": state.question,
                "citation_ids": ", ".join(citation_map.keys())
            })

            retry_answer = str(retry_result.content).strip()

            (
                retry_answer,
                retry_valid_ids,
                retry_invalid_ids
            ) = self._normalize_and_validate_citations(
                retry_answer,
                citation_map
            )

            if retry_valid_ids:
                answer = retry_answer
                valid_ids = retry_valid_ids
                invalid_ids = retry_invalid_ids

        # Final safety check.
        if not valid_ids and self.NO_ANSWER.lower() not in answer.lower():
            print(
                "WARNING: No valid citations after retry. "
                "Returning safe abstention."
            )
            answer = self.NO_ANSWER

        if valid_ids:
            print(f"\nValidated citation IDs: {valid_ids}")

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            relevant_docs=state.relevant_docs,
            is_relevant=True,
            answer=answer
        )
