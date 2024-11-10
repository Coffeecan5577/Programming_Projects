# import boto3
from langchain_ollama import OllamaEmbeddings
# from langchain_community.embeddings import BedrockEmbeddings


def get_embedding_function():
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")
#
    return embeddings
