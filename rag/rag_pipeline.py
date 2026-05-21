import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
load_dotenv()

# ── Initialize Embeddings (Free, No API needed) ───────────
def get_embeddings():
    """Use free HuggingFace embeddings — no API key needed."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# ── Convert DataFrame to Documents ───────────────────────
def dataframe_to_documents(df: pd.DataFrame) -> list:
    """Convert each row of dataframe into a LangChain Document."""
    documents = []
    for idx, row in df.iterrows():
        # Convert row to readable text
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
        doc = Document(
            page_content=row_text,
            metadata={"row_index": idx}
        )
        documents.append(doc)

    # Also add a summary document
    summary_text = f"""
Dataset Summary:
- Total Rows: {df.shape[0]}
- Total Columns: {df.shape[1]}
- Columns: {', '.join(df.columns.tolist())}
- Numeric columns: {', '.join(df.select_dtypes(include='number').columns.tolist())}
- Categorical columns: {', '.join(df.select_dtypes(include='object').columns.tolist())}

Statistical Summary:
{df.describe().to_string()}
    """
    documents.append(Document(
        page_content=summary_text,
        metadata={"row_index": -1, "type": "summary"}
    ))

    return documents

# ── Build Vector Store ────────────────────────────────────
def build_vectorstore(df: pd.DataFrame, persist_dir: str = "./chroma_db"):
    """Build and persist a ChromaDB vector store from dataframe."""
    print("🔨 Building vector store...")

    # Convert df to documents
    documents = dataframe_to_documents(df)

    # Split large documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(documents)

    # Get embeddings
    embeddings = get_embeddings()

    # Build ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"✅ Vector store built with {len(split_docs)} chunks!")
    return vectorstore

# ── Load Existing Vector Store ────────────────────────────
def load_vectorstore(persist_dir: str = "./chroma_db"):
    """Load an existing ChromaDB vector store."""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vectorstore

# ── RAG Query ─────────────────────────────────────────────
def rag_query(query: str, vectorstore, k: int = 5) -> str:
    """
    Retrieve relevant context from vector store for a query.
    Returns the most relevant text chunks.
    """
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        docs = retriever.invoke(query)
        
        # Combine retrieved chunks into context
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        return f"RAG retrieval error: {str(e)}"

# ── RAG-Enhanced Answer ───────────────────────────────────
def rag_answer(query: str, vectorstore) -> str:
    """
    Get a RAG-enhanced answer using retrieved context + LLM.
    """
    try:
        # Step 1: Retrieve relevant context
        context = rag_query(query, vectorstore)

        # Step 2: Build prompt with context
        prompt = f"""
You are QueryQuill 🪶, an expert data analyst.
Use the following retrieved data context to answer the question accurately.

Retrieved Context:
{context}

Question: {query}

Instructions:
- Answer based on the retrieved context
- Be specific with numbers and values from the data
- If the context doesn't contain enough info, say so
- Keep the answer clear and concise
"""

        # Step 3: Get LLM response
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"❌ RAG error: {str(e)}"