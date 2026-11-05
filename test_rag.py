
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


print("1. Initializing LLM...")
llm = Config.get_llm()

print("\n2. Loading PDF...")
processor = DocumentProcessor(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP
)

documents = processor.load_documents(["data/attention.pdf"])
chunks = processor.split_documents(documents)

print("Pages loaded:", len(documents))
print("Chunks created:", len(chunks))

print("\n3. Creating vector store...")
vector_store = VectorStore()
vector_store.create_vectorstore(chunks)

print("\n4. Building LangGraph...")
graph_builder = GraphBuilder(
    retriever=vector_store.get_retriever(),
    llm=llm
)

graph_builder.build()

print("\n5. Asking a question...")
question = "What is the purpose of multi-head attention?"
result = graph_builder.run(question)

print("\nQUESTION:", question)
print("\nANSWER:", result["answer"])

print("\nRETRIEVED DOCUMENTS:")
for i, doc in enumerate(result["retrieved_docs"], 1):
    print(f"\nDocument {i}")
    print("Page:", doc.metadata.get("page"))
    print(doc.page_content[:300])

print("\nRAG TEST COMPLETED!")
