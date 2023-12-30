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
