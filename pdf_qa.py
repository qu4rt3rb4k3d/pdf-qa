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
from langchain.document_loaders import PyPDFLoader
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
parser.add_argument('--path', type=str, default=None, help='path of PDF file')
parser.add_argument('--chunk-length', type=int, default=256, help='length of chunks (in tokens) to split the document into')
parser.add_argument('--num-chunks', type=int, default=8, help='number of chunks to use in a query')
parser.add_argument('--mmr', action='store_true', help='use maximal marginal relevance search')
args = parser.parse_args()

url = args.url
path = args.path
chunk_length = args.chunk_length
num_chunks = args.num_chunks
mmr = args.mmr

assert url != path, 'Need to provide either a URL or a path'

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

vectorstore = get_vectorstore(docs, chunk_length)

log_file = open('log.txt', 'a')
now = datetime.datetime.now()
log_file.write('Conversation started at ' + now.strftime('%Y-%m-%d %H:%M:%S') + ':\n')

message_log = []

while True:
    query = input('Q: ')
    if query == 'exit':
        break
    log_file.write('Q: ' + query + '\n')

    if mmr:
        chunks = vectorstore.max_marginal_relevance_search(query=query, k=num_chunks, fetch_k=num_chunks*4)
    else:
        chunks = vectorstore.similarity_search(query=query, k=num_chunks)
    chunks_string = str(reduce(lambda a, b: a + b, map(lambda c: [c.page_content], chunks)))
    system_message = '\nThe above are excerpts of a research paper, followed by previous questions & reponses. Do your best to answer the following question using them.'
    messages = []
    messages.append({'role': 'system', 'content': chunks_string})
    messages.extend(message_log)
    messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': query})

    response = dispatch_chat_completion(messages, model="gpt-4")
    message_log.append({'role': 'user', 'content': query})
    message_log.append({'role': 'assistant', 'content': response})
    print('\nA: ' + response + '\n')
    log_file.write('\nA: ' + response + '\n\n')

log_file.close()
