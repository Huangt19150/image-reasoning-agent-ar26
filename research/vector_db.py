import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

# TODO: This is only a backbone
# Pending what document would be useful for getting more valuable interpretations for the task

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def generate_persistent_vector_store(input_dir, persist_directory="db"):
    """
    This function is used to generate the persistent vector database
    Args:
            - input_dir (string) - The complete path of the input directory
            - persist_directory (string) - The complete path of the persistent directory
    """ 

    global db

    # Initialise the embedding function
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # If the persistent directory exists, then automatically initialise the vector database with the
    # embeddings from the persistent directory
    if os.path.exists(persist_directory):

        # Initialise the Chroma vector database
        db = Chroma(
            persist_directory=persist_directory, embedding_function=embedding_function
        )
        print("Caching successful")

    # If the persistent directory doesn't exist, then create it
    else:
        print("Creating persistent directory...")

        # Create the list of the paths of the PDF documents
        documents = []

        # Loop through all the PDF files in the folder
        for file in os.listdir(input_dir):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(input_dir, file)
                loader = PyPDFLoader(pdf_path)

                # Add the documents to the list
                documents.extend(loader.load())

        # Create Chroma vector database
        db = Chroma.from_documents(
            documents, embedding_function, persist_directory=persist_directory
        )

        print("Persistent directory created")


def query_db_with_llm(llm, query, persist_directory='db'):

    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

    # Initialize a retriever from the vector database
    retriever = db.as_retriever(search_type="similarity")

    # Create a Retrieval Q&A chain
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
        )

    result = qa({"query": query})

    return result['result']
