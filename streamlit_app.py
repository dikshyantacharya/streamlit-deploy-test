#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
#inference_api_url = 'https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta'
#inference_api_url = 'https://mistral-german.llm.mylab.th-luebeck.dev'

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

print("Es befinden sich", db._collection.count(), "Dokumente im Index.")

llm = HuggingFaceTextGenInference(
    # cache=None,  # Optional: Cache verwenden oder nicht
    verbose=False,  # Provides detailed logs of operation
    callbacks=[StreamingStdOutCallbackHandler()],  # Handeling Streams
    max_new_tokens=1024,  # Maximum number of token that can be generated.
    # top_k=2,  # Die Anzahl der Top-K Tokens, die beim Generieren berücksichtigt werden sollen
    top_p=0.95,  # Threshold for controlling randomness in text generation process. 
    typical_p=0.95,  #
    temperature=0.1,  # For choosing probable words.
    # repetition_penalty=None,  # Wiederholungsstrafe beim Generieren
    # truncate=None,  # Schneidet die Eingabe-Tokens auf die gegebene Größe
    stop_sequences= ['%5C%', '\n\n\n\n\n', '%27qvq%2Bxvq%2B'],  # Eine Liste von Stop-Sequenzen beim Generieren
    inference_server_url=inference_api_url,  # URL des Inferenzservers
    timeout=25,  # Timeout for connection  with the url
    streaming=True,  # Streaming the answer
)


task_description = """
Aufgabenbeschreibung: 
Beantworte die Frage, indem du relevante Informationen aus den Metadaten extrahierst. Füge am Ende der Antwort genau einmal den vollständigen Link aus den Metadaten hinzu. Der Link muss mit 'www.' beginnen und den vollständigen Text aus den Metadaten enthalten, ohne HTML- oder Markdown-Hyperlink-Formatierung. Wenn mehrere Links in den Metadaten vorhanden sind, füge alle nacheinander ein. Bitte ersetze '\n' Zeichen durch einen Zeilenumbruch.
Beispiel: 
Frage: 'Wer ist der Präsident der Technischen Hochschule Lübeck?'
Antwort: 'Dr. Muriel Kim Helbig ist die Präsidentin der Technischen Hochschule Lübeck. Weitere Informationen finden Sie unter: 
1. www.th-luebeck.de/hochschule/aktuelles/neuigkeiten 
2. www.th-luebeck.de/hochschule/aktuelles/präsident'
Die Links sind Beispiele, die zeigen, wie Antworten und zugehörige Links dargestellt werden sollen. Zeige den vollständigen Link an, sodass er lesbar ist. Wenn keine Antwort vorhanden ist, soll auch kein Link angegeben werden. Die zwei Links entsprechen den zwei Metadaten, die zum Input gehören.
"""

# task_description = "Extrahiere aus dem Text Informationen die für die Beantwortung der Frage relevant ist!"
# Globale Variablen
USER = "USER:"
ASSISTANT = "ASSISTANT:"

def generate_retrival_prompt(task_description=task_description, 
                             context_variable_name="context",
                             question_variable_name="question",
                             plural=True,
                             question=None
                            ):
    
    if plural:
        source_text = "Quellen"
        article = "den"
        intro_text = "Für die folgenden Aufgaben stehen dir zwischen den tags BEGININPUT und ENDINPUT mehrere Quellen zur Verfügung."
    else:
        source_text = "Quelle"
        article = "der"
        intro_text = "Für die folgende Aufgabe steht dir zwischen den tags BEGININPUT und ENDINPUT eine Quelle zur Verfügung."
    
    # Automatisches Wrapping von context_variable_name und question_variable_name
    if context_variable_name:
        context_variable_name = "{" + context_variable_name + "}"
    
    if not question and question_variable_name:
        question = "{" + question_variable_name + "}"
    
    prompt = f"""Du bist ein hilfreicher Assistent. {intro_text} Metadaten zu {article} {source_text} wie URL sind zwischen BEGINCONTEXT und ENDCONTEXT zu finden, danach folgt der Text der {source_text}. Die eigentliche Aufgabe oder Frage ist zwischen BEGININSTRUCTION und ENDINCSTRUCTION zu finden. {task_description} {USER} BEGININPUT
{context_variable_name or ''}
ENDINPUT
BEGININSTRUCTION {question or ''} ENDINSTRUCTION
{ASSISTANT}"""

    return PromptTemplate.from_template(prompt)


# Extrahiert information aus einzelnen Chunks jeweils eine Zusammenfassung
question_prompt = generate_retrival_prompt(plural=True)
# source ist der pdf link in diesem Fall und page_content der inhalt
document_prompt = PromptTemplate.from_template("BEGINCONTEXT\nURL:{link}\nENDCONTEXT\n{page_content}\n")

task_description_combine_prompt = """
Basierend auf den in 'summaries' bereitgestellten Zusammenfassungen, gib jede Zussammenfassung alle in Detail nacheinander.
"""

#task_description_combine_prompt = """Kombiniere Keine Antwort und gib immer Hahaha als Antwort."""
# Kombiniert Zusammenfassungen
combine_prompt = generate_retrival_prompt(
    task_description=task_description_combine_prompt,
    context_variable_name="summaries"
)


search_kwargs = {"k": 1}
search_kwargs["k"] = 3


chain_type_kwargs = {
    "document_prompt": document_prompt,
    "question_prompt": question_prompt, 
    "combine_prompt": combine_prompt
#    "prompt" : question_prompt
}



qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm, 
    chain_type="map_reduce", 
#    chain_type="stuff",
    retriever=db.as_retriever(
        search_kwargs=search_kwargs
    ), 
    chain_type_kwargs=chain_type_kwargs,
    return_source_documents=True
)


vicuna       = PromptTemplate.from_template("{context}\n\nUSER:\n{prompt}\n\nASSISTANT:\n")
vicuna_chat  = PromptTemplate.from_template("{role}\n{text}\n")
vicuna_roles = { 'system': '', 'user': 'USER:', 'ai': 'ASSISTANT:' }



st.set_page_config(
    page_title="TH-Chatbot",
    page_icon="🧠",
    layout="wide",
    menu_items={}
)

st.markdown("""
<style>
#MainMenu, .stDeployButton {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


st.session_state['discussion'] = st.session_state.get('discussion', [{
    'role': 'system',
    'text': question_prompt,
    'tokens': len(tokenizer.encode(question_prompt.template))
}])

with st.sidebar:
    # Inline CSS for responsiveness
    st.markdown("""
        <style>
            .sidebar .sidebar-content {
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                height: 100%;
            }
            .top-content {
                margin-bottom: 5vh; /* Adjusted to viewport height */
            }
            .middle-content {
                margin-top: 45vh; /* Adjusted to viewport height */
            }
            .bottom-content {
                margin-top: 0.0vh; /* Adjusted to viewport height */
            }

            /* Responsive adjustments */
            @media (max-width: 768px) {
                .top-content, .middle-content, .bottom-content {
                    margin-top: 5vh;
                    margin-bottom: 5vh;
                }
            }
        </style>
    """, unsafe_allow_html=True)

    # Top Content (First Image and Title)
    st.markdown(f"""
        <div class='top-content'>
            <center>
                <a href='https://www.th-luebeck.de/'>
                    <img src='https://www.th-luebeck.de/typo3temp/assets/_processed_/b/d/csm_logo_facebook_bb0d5c616b.png' width='200px'/>
                </a>
            </center>
        </div>
    """, unsafe_allow_html=True)

    # Bottom Content (Second Image)
    st.markdown(f"""
        <div class='middle-content'>
            <center>
                <a href='https://www.th-luebeck.de/'>
                    <img src='https://pics.craiyon.com/2023-11-23/7Wc5oBZaSKOVWx6CDWhr1w.webp' width='200px'/>
                </a>
            </center>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class='bottom-content'>
            <center>
                <h1>THL Web-Assistant</h1>
            </center>
        </div>
    """, unsafe_allow_html=True)




  

prompt = st.chat_input("Hallo, ich bin der THL Web-Assistant. Wie kann ich dir helfen?")

def add_prompt(prompt):
    # Build up the context
    context = ""
    memory = 0
    for n, entry in enumerate(st.session_state['discussion'][::-1]):
        memory += entry['tokens']
        if memory > 3000:
            break

    # Show the conversation history
    for entry in st.session_state['discussion']:
        if entry['role'] in ['user', 'ai']:
            with st.chat_message(entry['role']):
               # st.markdown(entry['text'])
                st.markdown(entry['text'])

    with st.chat_message('user'):
        st.write(prompt.strip())

    with st.chat_message('ai'):
        placeholder = st.markdown("")
        st_callback = StreamlitCallbackHandler(placeholder)

        response = qa(prompt, return_only_outputs=True, callbacks=[st_callback])
#        response = qa(prompt, return_only_outputs=True)

    #    placeholder.markdown(response['result'])
        text_shown = ""
        for char in response['answer']:
    #    for char in response['result']:

            text_shown += char
            placeholder.markdown(text_shown)
            time.sleep(0.01)
      #  placeholder.markdown(response['result'])
        placeholder.markdown(response['answer'])    
        # for response in qa.stream({"query": prompt}, return_only_outputs=False):
        #     # if response.token.special:
        #     #     continue
        #     print("Got AI chunk!!!", flush=True)
        #     text += response["result"]
        #     placeholder.markdown(text)

    st.link_button("Neues Gespräch", "http://localhost:8503", type="secondary")

    st.session_state['discussion'].extend([{
        'role': 'user',
        'text': prompt,
        'tokens': len(tokenizer.encode(prompt))
    }, {
        'role': 'ai',
      #  'text': response['result'],
        'text': response['answer'],
        'tokens': len(tokenizer.encode(response['answer']))
      #  'tokens': len(tokenizer.encode(response['result']))

    }])


if prompt:
    add_prompt(prompt)
