import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.chains import RetrievalQA

DATA_FOLDER = "data"
CHROMA_DB = "db"
EMBED_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.5-pro"

# Load all PDFs from the data directory
def load_docs():
    docs = []
    for f in os.listdir(DATA_FOLDER):
        if f.endswith(".pdf"):
            path = os.path.join(DATA_FOLDER, f)
            print(f"[INFO] Loading: {path}")
            docs.extend(PyPDFLoader(path).load())
    # print(f"[INFO] Total documents loaded: {len(docs)}")
    return docs

# Prepare and embed the documents into a Chroma vectorstore
def prepare_vectorstore():
    documents = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    # print(f"[INFO] Total chunks created: {len(chunks)}")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB
    )
    # vectordb.persist()
    # print("[INFO] Vectorstore created and persisted.")
    return vectordb

# Retrieve the QA chain
def get_chain():
    # Check whether vector DB exists and is not empty
    db_exists = os.path.exists(CHROMA_DB) and len(os.listdir(CHROMA_DB)) > 0

    if db_exists:
        # print("[INFO] Loading existing vectorstore...")
        vectordb = Chroma(
            persist_directory=CHROMA_DB,
            embedding_function=GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        )
    else:
        # print("[INFO] No vectorstore found. Creating new one...")
        vectordb = prepare_vectorstore()

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)

    prompt = PromptTemplate.from_template("""
You are a tourism assistant. Use the context to answer questions about tourism or Nublo's privacy policy. If context isn't helpful, use your own knowledge.
Do not tell how you are answering , based on context or whatever.
Context:
{context}
Question: {question}
Answer:
""")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

# start RAG tool 
def get_rag_tool():
    return Tool.from_function(
        func=lambda q: get_chain().invoke({"query": q})["result"],
        name="Tourism_RAG_Tool",
        description="Answer FAQs/ questions about tourism or Nublo's privacy policy using uploaded documents."
    )
