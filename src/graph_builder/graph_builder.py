
from langgraph.graph import StateGraph, END

from src.state.rag_state import RAGState
from src.node.reactnode import RAGNodes


class GraphBuilder:

    def __init__(self, retriever, llm):

        self.nodes = RAGNodes(retriever, llm)
        self.graph = None

    # Decide whether to generate an answer or abstain
    def route_after_grading(self, state: RAGState):

        if state.is_relevant:
            return "generate"

        return "no_answer"

    # Build LangGraph workflow
    def build(self):

        builder = StateGraph(RAGState)

        # Register nodes
        builder.add_node(
            "retriever",
            self.nodes.retrieve_docs
        )

        builder.add_node(
            "grader",
            self.nodes.grade_documents
        )

        builder.add_node(
            "responder",
            self.nodes.generate_answer
        )

        # Node for questions with no relevant documents
        def no_answer_node(state: RAGState):

            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                relevant_docs=[],
                is_relevant=False,
                answer=(
                    "The provided documents do not contain "
                    "sufficient information to answer this question."
                )
            )

        builder.add_node("no_answer", no_answer_node)

        # Define workflow
        builder.set_entry_point("retriever")

        builder.add_edge("retriever", "grader")

        # Conditional routing
        builder.add_conditional_edges(
            "grader",
            self.route_after_grading,
            {
                "generate": "responder",
                "no_answer": "no_answer"
            }
        )

        builder.add_edge("responder", END)
        builder.add_edge("no_answer", END)

        self.graph = builder.compile()

        return self.graph

    # Execute workflow
    def run(self, question: str):

        if self.graph is None:
            self.build()

        initial_state = RAGState(
            question=question
        )

        return self.graph.invoke(initial_state)
