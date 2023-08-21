from semantic_split.splitters.SpacySentenceSplitter import SpacySentenceSplitter
from semantic_split.splitters.SimilarSentenceSplitter import SimilarSentenceSplitter
from semantic_split.SentenceSimilarity import SentenceTransformersSimilarity
import weaviate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate, Chroma, FAISS, Qdrant
from langchain.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.memory import ConversationSummaryBufferMemory, ZepMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.retrievers import ZepRetriever
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.document_loaders import PyMuPDFLoader,PDFMinerLoader,PDFPlumberLoader
from transformers import pipeline, AutoTokenizer
from langchain import OpenAI, LLMChain
from uuid import uuid4
import gradio as gr
from langchain.docstore.document import Document
import json
import spacy

nlp = spacy.load("en_core_web_md")

model = SentenceTransformersSimilarity()
sentence_splitter = SpacySentenceSplitter()
splitter = SimilarSentenceSplitter(model, sentence_splitter)


def sliding_window_segmentation(text, window_size=3, overlap=2):
    # Split the text into sentences using a simple regex pattern
    doc = nlp(escape_characters(text))
    sentences = [sent.text for sent in doc.sents]

    # Validate window_size and overlap
    if window_size <= 0 or overlap < 0 or overlap >= window_size:
        raise ValueError("Invalid window_size or overlap parameters.")

    # Initialize lists to store the chunks and the current window
    chunks = []
    window = []

    # Loop through sentences to create chunks
    for i, sentence in enumerate(sentences):
        window.append(sentence)

        # Check if the window size has been reached or it's the last sentence
        if len(window) == window_size or i == len(sentences) - 1:
            chunk = ' '.join(window)
            chunks.append(chunk)

            # Move the window by the specified overlap
            window = window[window_size - overlap:]

    return chunks



model_name = "Gladiator/microsoft-deberta-v3-large_ner_conll2003"
pipe = pipeline("token-classification", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def ner(text):
    results = pipe(text)

    # Convert IOB2 format to the custom format of the API response
    entities = []
    current_entity = None
    for token_info in results:
        if token_info["entity"].startswith("B-"):
            # Start of a new entity
            current_entity = {
                "entity_group": token_info["entity"][2:],  # Remove the "B-" prefix
                "word": token_info["word"],
                "start": token_info["start"],
                "end": token_info["end"],
            }
            entities.append(current_entity)
        elif token_info["entity"].startswith("I-") and current_entity:
            # Inside an entity, append the word to the current entity
            current_entity["word"] += token_info["word"]
            current_entity["end"] = token_info["end"]
        else:
            # Outside an entity or non-entity token, reset the current entity
            current_entity = None

    # Decode word pieces back to original tokens
    for entity in entities:
        word_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(entity["word"], add_special_tokens=False))
        entity["word"] = tokenizer.convert_tokens_to_string(word_tokens)

    # Remove the "score" key from the results
    for entity in entities:
        entity.pop("score", None)

    return {"entities": entities}

# hf ner kwextraction
import unicodedata,re

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def escape_characters(text):
    words = text.split(' ')
    new_list = []
    for word in words:
        if "\u2019" not in word:
            normalized_text = unicodedata.normalize("NFKD", word)
            stripped_accents = strip_accents(normalized_text)
            cleaned = re.sub(r"[^'\x00-\x7f]", r' ', stripped_accents).strip()
            new_list.append(cleaned)
        else:
            new_list.append(word)

    output = ' '.join(new_list)
    return output.replace('\u2019', "'")



# PyMuPDFLoader e5 passage and query
 
# model_name = "msmarco-distilbert-base-v4"

model_name = "intfloat/e5-large-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)



ZEP_API_URL = "http://localhost:8000"
session_id = str(uuid4())

import fitz

import os

os.environ["OPENAI_API_KEY"] = "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"
# os.environ["WEAVIATE_URL"] = "http://localhost:8080/"

# WEAVIATE_URL = os.getenv("WEAVIATE_URL")

# client = weaviate.Client(
#     url=WEAVIATE_URL,
#     additional_headers={
#         "X-Openai-Api-Key": os.getenv("OPENAI_API_KEY"),
#     },
# )

zep_retriever = ZepRetriever(
    session_id=session_id,
    url=ZEP_API_URL,
    top_k = 5
)

zep_memory = ZepMemory(
    session_id=session_id,
    url=ZEP_API_URL,
    # return_messages=True,
    # output_key="answer",
    # output_key='answer',
    memory_key="chat_history",
    input_key="question"
)

def split_text(text):
    # split the text into chunks of size 200
    text_splitter = RecursiveCharacterTextSplitter(
            # separator='\n', 
            chunk_size=1000,
            separators=[" ", ",", "\n"],
            chunk_overlap=1000 * 0.3
        )

    texts = text_splitter.split_text(text)
    return texts


text_splitter = RecursiveCharacterTextSplitter(
        # separator='\n', 
        chunk_size=1000,
        separators=[" ", ",", "\n"],
        chunk_overlap=1000 * 0.3
    )

text_splitter2 = RecursiveCharacterTextSplitter(
        # separator='\n', 
        chunk_size=500,
        separators=[" ", ",", "\n"],
        chunk_overlap=500 * 0.1
    )

# import spacy
# import en_core_web_md

import unicodedata,re
# nlp = en_core_web_md.load()

import requests

url = "http://0.0.0.0:5000/"

def get_keywords(text):

    payload = json.dumps({
    "text": text,
    "lang": "en",
    "n": 10
    })
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    keywords, scores, relevant_keywords = response.json()
    if keywords != None:

        # Pair each keyword with its corresponding score
        keyword_score_pairs = list(zip(keywords, scores))

        # Sort the pairs based on the scores (in descending order)
        sorted_keyword_score_pairs = sorted(keyword_score_pairs, key=lambda x: x[1], reverse=True)

        # Extract the sorted keywords and scores separately
        sorted_keywords, sorted_scores = zip(*sorted_keyword_score_pairs)

        return {"keywords":list(sorted_keywords[:8])}
    else:
        return {"keywords":["passage"]}

loader = PyMuPDFLoader("/Users/vagif/Desktop/chat_assistant/lec23.pdf")
data = loader.load()


documents = text_splitter.split_documents(data)


texts = " ".join([escape_characters(doc.page_content) for doc in documents])
metadatas = [doc.metadata for doc in documents]

model = SentenceTransformersSimilarity()
sentence_splitter = SpacySentenceSplitter()
splitter = SimilarSentenceSplitter(model, sentence_splitter)

chunks = []

objs = sliding_window_segmentation(texts,window_size=5)
for o in objs:
   chunks.append(o)

final_chunks = []

for c in chunks:
    res = splitter.split(c)
    for r in res:
        final_chunks.append(" ".join(r))


# exit()
final_split = []


for c in final_chunks:
    # extracted_keywords = get_keywords(c)
    new_data = "passage: " + c
    c = new_data
    # extracted_entities = ner(c)
    # if 'error' in extracted_entities['entities']:
    #     print("The 'error' key is present.")
    #     exit()
    doc =  Document(page_content=c, metadata={})
    doc.metadata.update(metadatas[0])
    # doc.metadata.update(extracted_entities)
    # doc.metadata.update(extracted_keywords)
    final_split.append(doc)

print(final_split[0].metadata)
# doc = fitz.open("./lec23.pdf")
# text = ''
# for page in doc:
#     text_page = page.get_text()
#     text += text_page


# from qdrant_client import QdrantClient

# client = QdrantClient(host="localhost", port=6333)


# texts = split_text(text)

# texts = ["passage: " + string for string in texts]

# print(texts)

# db = FAISS.from_texts(texts, hf)

from langchain.vectorstores import Milvus

db = Milvus.from_documents(
    final_split,
    hf,
    connection_args={"host": "127.0.0.1", "port": "19530"},
    collection_name="shayo",
    index_params={
        "metric_type": "IP",
        "index_type": "FLAT",
    },
    search_params={"metric_type": "IP"},
)


from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.merger_retriever import MergerRetriever


# retriever = db.as_retriever(search_type="mmr",search_kwargs={'k':4})
retriever1 = db.as_retriever(search_kwargs={'k':5})

bm25_retriever = BM25Retriever.from_documents(final_split)
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever1], weights=[0.5, 0.5])



def ask(query):
    # self.find_from_history(query)
    # result = self.qa({"question":query,"chat_history":self.chat_history}) # given chat history, find a relevant answer to the given question
    result = qa({"question":query}) # given chat history, find a relevant answer to the given question
    return result['answer']

# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot()
#     msg = gr.Textbox()
#     clear = gr.ClearButton([msg, chatbot])

#     def respond(message, ch):
#         bot_message = ask(message)
#         ch.append((message, bot_message))
#         return "", ch

#     msg.submit(respond, [msg, chatbot], [msg, chatbot])

# demo.launch()

template = """
You are a Q&A answering assistant. Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer a given question in structured paragraphs:
------
<ctx>
{context}
</ctx>
------
<hs>
{chat_history}
</hs>
------
Question: {question}
Answer:
"""


from langchain import PromptTemplate


prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=template,
)

llm = ChatOpenAI(
    model_name='gpt-4',
    temperature=0
)

import openai

def clf(text):
    res = openai.Completion.create(
        model="ada:ft-upword-2023-08-10-17-04-01",
        temperature=1,
        max_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        prompt="[Question]: {t}\n[Classification]:".format(t=text))
    
    return res.choices[0].text


general_system_template = """You are a Q&A answering assistant. Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer a given question in structured paragraphs:
        ------
        <ctx>
        {context}
        </ctx>
        ------
        <hs>
        {history}
        </hs>
        ------
        """
general_user_template = "```{question}```"
messages = [
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template(general_user_template)
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

zep_memory = ZepMemory(
    session_id=session_id,
    url=ZEP_API_URL,
    return_messages=True,
    # output_key="answer",
    memory_key="history",
    input_key="question"
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=ensemble_retriever,
    chain_type_kwargs={"memory":zep_memory,"prompt": qa_prompt}
)

print(qa.run({"query":"query: What are the two principles of justice?"}))
print("----------------------------------")
print(qa.run({"query":"query: What are the three values?"}))
print("----------------------------------")
print(qa.run({"query":"query: What was the last question I asked about?"}))
print("----------------------------------")
print(qa.run({"query":"query: Explain the third value"}))
print("----------------------------------")
print(qa.run({"query":"query: Please extend the answer of the previous question I asked you about"}))
