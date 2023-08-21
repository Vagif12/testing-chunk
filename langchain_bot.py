from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    './Langchain/data/', # my local directory
    glob='**/*.pdf',     # we only get pdfs
    show_progress=True
)
docs = loader.load()


from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=0
)
docs_split = text_splitter.split_documents(docs)

import os

PINECONE_API_KEY = ... # find at app.pinecone.io
PINECONE_ENV = ...     # next to api key in console
OPENAI_API_KEY = ...   # found at platform.openai.com/account/api-keys

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

import pinecone 
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

# we use the openAI embedding model
embeddings = OpenAIEmbeddings()
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

doc_db = Pinecone.from_documents(
    docs_split, 
    embeddings, 
    index_name='langchain-demo'
)


query = "What were the most important events for Google in 2021?"
search_docs = doc_db.similarity_search(query)
print(search_docs)

from langchain import OpenAI
llm = OpenAI()

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI()

from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff',
    retriever=doc_db.as_retriever(),
)

query = "What were the earnings in 2022?"
result = qa.run(query)

print(result)