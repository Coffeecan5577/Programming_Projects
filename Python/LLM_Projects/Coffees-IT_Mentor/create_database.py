from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import ollama
import os
import shutil

CHROMA_PATH = "/home/coffeecan/Git-Repos/Programming_Projects/Python/Chroma_Training_Database"
DATA_PATH = "/home/coffeecan/Git-Repos/Programming_Projects/Python/LLM_Projects/Coffees-IT_Mentor/Training_Data"

def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md, *.pdf", show_progress=True, use_multithreading=True)
    documents = loader.lazy_load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(tuple(documents))} documents into {len(chunks)} chunks.")

    for document in documents:
        print(document.page_content)
        print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB fromm the documents.
    db = Chroma.from_documents(
        chunks, OllamaEmbeddings(model="nomic-embed-text",), persist_directory=CHROMA_PATH
        )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()