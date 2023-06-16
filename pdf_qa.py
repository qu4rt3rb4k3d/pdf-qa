import openai
import tiktoken
import json
import time
import datetime
import sys
import os
import argparse
import requests
import re
import random
from functools import reduce
from pypdf import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def dispatch_chat_completion(messages, model, temperature=0):
    while True:
        try:
            completion = openai.ChatCompletion.create(messages=messages, model=model, temperature=temperature)
            break
        except Exception as e:
            print(repr(e))
            time.sleep(random.uniform(5, 20))
    return completion.choices[0].message["content"]

def get_vectorstore(docs, chunk_length):
    tokenizer = tiktoken.encoding_for_model('gpt-4')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_length, chunk_overlap=0, length_function=lambda s: len(tokenizer.encode(s)))
    chunks = text_splitter.split_documents(docs)
    return Chroma.from_documents(chunks, OpenAIEmbeddings())

parser = argparse.ArgumentParser()
parser.add_argument('--url', type=str, default=None, help='URL of PDF file')
parser.add_argument('--file', type=str, default=None, help='path of PDF file')
parser.add_argument('--dir', type=str, default=None, help='directory of PDF files')
parser.add_argument('--chunk-length', type=int, default=256, help='length of chunks (in tokens) to split the document into')
parser.add_argument('--num-chunks', type=int, default=8, help='number of chunks to use in a query')
parser.add_argument('--mmr', action='store_true', help='use maximal marginal relevance search')
parser.add_argument('--no-chunking', action='store_true', help='disable chunking and vectorstore')
parser.add_argument('--model', type=str, default='gpt-4', help='model to use for QA')
args = parser.parse_args()

url = args.url
file_path = args.file
dir_path = args.dir
chunk_length = args.chunk_length
num_chunks = args.num_chunks
mmr = args.mmr
chunking = not args.no_chunking
qa_model = args.model

assert sum(map(lambda a: a is not None, [url, file_path, dir_path])) == 1, 'need to provide a URL, a file path, or a directory path'
assert chunking or not dir_path, 'loading multiple documents with chunking disabled is not currently supported'

config_file = open('config.json')
config = json.loads(config_file.read())
openai.api_key = config['OpenAI Key']
os.environ['OPENAI_API_KEY'] = config['OpenAI Key']
config_file.close()

if url:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    path = '/tmp/paper.pdf'
    if response.headers['Content-Type'] == 'application/pdf':
        with open(path, 'wb') as f:
            f.write(response.content)
    else:
        print('The URL did not return a PDF file')
        exit()
    loader = PyPDFLoader(path)
    docs = loader.load()
elif file_path:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
elif dir_path:
    loader = PyPDFDirectoryLoader(dir_path)
    docs = loader.load()

if chunking:
    vectorstore = get_vectorstore(docs, chunk_length)
else:
    context_string = reduce(lambda a, b: a + b, map(lambda c: c.page_content, docs))

log_file = open('log.txt', 'a')
now = datetime.datetime.now()
log_file.write('Conversation started at ' + now.strftime('%Y-%m-%d %H:%M:%S') + ':\n')

message_log = []

while True:
    query = input('Q: ')
    if query == 'exit':
        break
    log_file.write('Q: ' + query + '\n')

    if chunking:
        if mmr:
            chunks = vectorstore.max_marginal_relevance_search(query=query, k=num_chunks, fetch_k=num_chunks*4)
        else:
            chunks = vectorstore.similarity_search(query=query, k=num_chunks)
        context_string = str(reduce(lambda a, b: a + b, map(lambda c: [c.page_content], chunks)))
        #chunks_string = str(reduce(lambda a, b: a + b, map(lambda c: str(c), chunks)))
        system_message = 'The above are excerpts from one or more research papers, followed by previous questions and reponses. Use this to answer the following question.'
    else:
        system_message = 'The above is a research paper, followed by previous questions and responses. Use this to answer the following question.'

    messages = []
    messages.append({'role': 'user', 'content': context_string})
    messages.extend(message_log)
    messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': query})

    response = dispatch_chat_completion(messages, model=qa_model)
    message_log.append({'role': 'user', 'content': query})
    message_log.append({'role': 'assistant', 'content': response})
    print('\nA: ' + response + '\n')
    log_file.write('\nA: ' + response + '\n\n')

log_file.close()
