import streamlit as st

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


@st.cache_resource
def initialize_rag():

    # 1. Initialize local LLM
    llm = Config.get_llm()

    # 2. Load and split documents
    processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )

    documents = processor.process_sources(["data/attention.pdf"])

    # 3. Create vector store
    vector_store = VectorStore()
    vector_store.create_vectorstore(documents)

    # 4. Build LangGraph
    graph_builder = GraphBuilder(
        retriever=vector_store.get_retriever(),
        llm=llm
    )

    graph_builder.build()

    return graph_builder


# Streamlit interface
st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG Document Assistant")
st.caption("Ask questions about your uploaded knowledge base.")

with st.spinner("Initializing RAG system..."):
    rag_system = initialize_rag()

st.success("RAG system is ready!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user question
question = st.chat_input("Ask a question about the document...")

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents and generating answer..."):

            result = rag_system.run(question)

            answer = result["answer"]

            st.markdown(answer)

            with st.expander("View retrieved document passages"):
                for i, doc in enumerate(result["retrieved_docs"], 1):
                    st.markdown(f"**Passage {i}**")
                    st.write("Page:", doc.metadata.get("page"))
                    st.write(doc.page_content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })