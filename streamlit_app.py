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

