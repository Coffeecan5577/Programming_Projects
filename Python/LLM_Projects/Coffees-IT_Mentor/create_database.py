import ollama
import os
import shutil
import logging
import tqdm
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# Define constants for paths
CHROMA_PATH = "/home/coffeecan/Git-Repos/Programming_Projects/Python/LLM_Projects/Coffees-IT_Mentor/Chroma_Training_Database"
DATA_PATH = "/home/coffeecan/Git-Repos/Programming_Projects/Python/LLM_Projects/Coffees-IT_Mentor/Training_Data/"
# OBSIDIAN_META_EXCLUSION_PATH = "/home/coffeecan/Git-Repos/Coffees-Obsidian-Vaults/Coffee's Vault/99 - Meta"

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_directory(DATA_PATH):
    logger.info("Printing documents in directory:")
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        try:
            with open(filepath, 'rb') as file:
                logger.info(f"File: {filename}, Size: {os.path.getsize(filepath)}")
                # print the contents of the file (you can use a debugger or inspect tool for this)
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")

def load_documents():
    try:
        logger.info("Loading documents from %s", DATA_PATH)
        loader = DirectoryLoader(DATA_PATH, glob="**/[!.]*", show_progress=True, use_multithreading=True, recursive=True, exclude="**/[/home/coffeecan/Git-Repos/Coffees-Obsidian-Vaults/Coffee's Vault/99 - Meta, /home/coffeecan/Git-Repos/Coffees-Obsidian-Vaults/Coffee's Vault/.obsidian, /home/coffeecan/Git-Repos/Coffees-Obsidian-Vaults/Coffee's Vault/.trash ]*", )
        documents = loader.load()
        logger.info("Loaded %d documents:", len(documents))
        # print_documents(DATA_PATH)  # <--- Add this line
        return documents
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        raise

def validate_input_documents(documents):
    try:
        logger.info("Validating input documents")
        if not all(isinstance(document, Document) for document in documents):
            logger.error("Invalid document format. Only .md and .pdf files are supported.")
            return None
        return documents
    except Exception as e:
        logger.error(f"Failed to validate input documents: {e}")
        raise

def split_text(documents):
    try:
        logger.info("Splitting text into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"],
            chunk_size=300,
            chunk_overlap=10,  # Reduced overlap to make more efficient splits
            length_function=len,
            add_start_index=False,
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        if len(chunks) < 10:
            logger.warning("Not enough chunks to display metadata.")
        else:
            document = chunks[100]
            logger.info("Displaying metadata of the first chunk")
            logger.info(document.page_content)
            logger.info(document.metadata)

        return chunks
    except Exception as e:
        error_message = f"Failed to split text: {e}. Please check the documents or the splitting strategy."
        logger.error(error_message)
        raise ValueError(error_message)

def save_to_chroma(chunks):
    logger.info("Saving documents to Chroma database at %s", CHROMA_PATH)
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OllamaEmbeddings(model="nomic-embed-text",), persist_directory=CHROMA_PATH
    )

    retriever = db.as_retriever(search_kwargs={"k": 10})
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
    try:
        documents = load_documents()
        if documents is None:
            return

        documents = validate_input_documents(documents)
        if documents is None:
            return

        chunks = split_text(documents)
        save_to_chroma(chunks)
    except Exception as e:
        logger.error(f"Failed to generate data store: {e}")

if __name__ == "__main__":
    generate_data_store()
