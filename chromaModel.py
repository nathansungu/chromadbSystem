from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def ChromaSetup(query:str):
    loader = PyPDFLoader("./Documents/omlprojectproposal.pdf") 
    documents = loader.load()

    # 2. Split into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # 3. Embed and Store in ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db" 
    )

    # generate response
    response = db.similarity_search(query, k=1)

    print(f"Added {len(chunks)} chunks to ChromaDB")
    return response   


