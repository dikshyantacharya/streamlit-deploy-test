__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import time
import chromadb
from langchain.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import HuggingFaceTextGenInference
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from tokenizers import Tokenizer
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer, util
import numpy as np
# Database 

tokenizer = Tokenizer.from_pretrained("bert-base-cased")

DB_PATH = os.environ.get("DB_PATH", os.getcwd() + "/data-par_mul-6000-optimized/chromadb")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "Dikshyant-Acharya")
inference_api_url = 'https://em-german-70b.llm.mylab.th-luebeck.dev'

persistent_client = chromadb.PersistentClient(path=DB_PATH)
collection = persistent_client.get_or_create_collection(COLLECTION_NAME)

model = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        # Initialize your sentence transformer model here
        self.model = SentenceTransformer(model)

    def embed_documents(self, texts):
        # Generate embeddings using the sentence transformer model
        embeddings = self.model.encode(texts, show_progress_bar= True)

        # Convert numpy array to a list of lists
        embeddings_list = embeddings.tolist()

        return embeddings_list

    def embed_query(self, query):
        # Embed a single query string
        embedding = self.model.encode([query], show_progress_bar=True)

        # Flatten the embedding if it's not already a 1-dimensional list
        if isinstance(embedding, list):
            # If it's a list of lists, flatten it
            if all(isinstance(elem, list) for elem in embedding):
                embedding = [item for sublist in embedding for item in sublist]
        elif isinstance(embedding, np.ndarray):
            # If it's an ndarray, convert it to a flat list
            embedding = embedding.flatten().tolist()

        return embedding


    def __call__(self, input: Documents) -> Embeddings:
        # Convert input documents to a list of strings
        document_texts = [doc.text for doc in input]

        # Call embed_documents
        return self.embed_documents(document_texts)


embedding_dimension = 768
db = Chroma(
    client=persistent_client,
    collection_name=COLLECTION_NAME,
    embedding_function=MyEmbeddingFunction(),
        collection_metadata={"hnsw:space": "cosine", "dimension": embedding_dimension}
)

#
