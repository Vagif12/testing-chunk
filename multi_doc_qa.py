# -*- coding: utf-8 -*-
"""multi_doc_qa.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10vhji3FPOAm43zAvjOF4dlgxAeqIkHDx
"""

! pip install llama-index nltk milvus pymilvus langchain python-dotenv openai

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")

from llama_index import (
    GPTVectorStoreIndex,
    GPTSimpleKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    StorageContext
)
from langchain.llms.openai import OpenAIChat

import os
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from llama_index.vector_stores import MilvusVectorStore
from milvus import default_server

default_server.start()
vector_store = MilvusVectorStore(
   host = "127.0.0.1",
   port = default_server.listen_port
)

wiki_titles = ["Toronto", "Seattle", "San Francisco", "Chicago", "Boston", "Washington, D.C.", "Cambridge, Massachusetts", "Houston"]

from pathlib import Path

import requests
for title in wiki_titles:
    response = requests.get(
        'https://en.wikipedia.org/w/api.php',
        params={
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            # 'exintro': True,
            'explaintext': True,
        }
    ).json()
    page = next(iter(response['query']['pages'].values()))
    wiki_text = page['extract']

    data_path = Path('data')
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", 'w') as fp:
        fp.write(wiki_text)

# Load all wiki documents
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(input_files=[f"data/{wiki_title}.txt"]).load_data()

llm_predictor_chatgpt = LLMPredictor(llm=OpenAIChat(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build city document index
city_indices = {}
index_summaries = {}
for wiki_title in wiki_titles:
    city_indices[wiki_title] = GPTVectorStoreIndex.from_documents(city_docs[wiki_title], service_context=service_context, storage_context=storage_context)
    # set summary text for city
    index_summaries[wiki_title] = f"Wikipedia articles about {wiki_title}"

from llama_index.indices.composability import ComposableGraph

graph = ComposableGraph.from_indices(
    GPTSimpleKeywordTableIndex,
    [index for _, index in city_indices.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50
)

from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
decompose_transform = DecomposeQueryTransform(
    llm_predictor_chatgpt, verbose=True
)

from llama_index.query_engine.transform_query_engine import TransformQueryEngine
custom_query_engines = {}

for index in city_indices.values():
    query_engine = index.as_query_engine(service_context=service_context)
    transform_extra_info = {'index_summary': index.index_struct.summary}
    tranformed_query_engine = TransformQueryEngine(query_engine, decompose_transform, transform_extra_info=transform_extra_info)
    custom_query_engines[index.index_id] = tranformed_query_engine

custom_query_engines[graph.root_index.index_id] = graph.root_index.as_query_engine(
    retriever_mode='simple',
    response_mode='tree_summarize',
    service_context=service_context
)

query_engine_decompose = graph.as_query_engine(
    custom_query_engines=custom_query_engines,)

response_chatgpt = query_engine_decompose.query(
    "Compare and contrast the airports in Seattle, Houston, and Toronto. "
)

print(str(response_chatgpt))

custom_query_engines = {}
for index in city_indices.values():
    query_engine = index.as_query_engine(service_context=service_context)
    custom_query_engines[index.index_id] = query_engine

custom_query_engines[graph.root_index.index_id] = graph.root_index.as_query_engine(
    retriever_mode='simple',
    response_mode='tree_summarize',
    service_context=service_context
)

query_engine = graph.as_query_engine(
    custom_query_engines=custom_query_engines,
)

response_chatgpt = query_engine.query(
    "Compare and contrast the airports in Seattle, Houston, and Toronto. "
)
str(response_chatgpt)

response_chatgpt = query_engine_decompose.query(
    "Compare and contrast the sports environment of Houston and Boston. "
)

str(response_chatgpt)
