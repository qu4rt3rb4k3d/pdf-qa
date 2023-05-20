import openai
import tiktoken
import json
import time
import datetime
import sys
import os
import asyncio
import concurrent.futures
import argparse
import requests
import re
import random
from typing import Any
from functools import reduce
from pypdf import PdfReader

def dispatch_chat_completion(messages, model):
    while True:
        try:
            completion = openai.ChatCompletion.create(messages=messages, model=model)
            break
        except Exception as e:
            print(repr(e))
            time.sleep(random.uniform(5, 20))
    return completion.choices[0].message["content"]

def preprocess_text(raw_text, keep_refs):
    if keep_refs:
        preprocess_prompt = "Please rewrite the above paper excerpt keeping only the main text and references section. In the references section, keep only the titles of the referenced documents."
        preprocess_model = "gpt-4"
    else:
        preprocess_prompt = "Please rewrite the above paper excerpt keeping only the main text."
        preprocess_model = "gpt-3.5-turbo"

    tokenizer = tiktoken.encoding_for_model(preprocess_model)
    tokenized_text = tokenizer.encode(raw_text)

    chunks = []
    for start in range(0, len(tokenized_text), 2000):
        end = min(len(tokenized_text), start+2000)
        chunks.append(tokenizer.decode(tokenized_text[start:end]))

    messages_list = []
    for chunk in chunks:
        messages_list.append([
            {"role": "user", "content": chunk},
            {"role": "user", "content": preprocess_prompt}
        ])

    preprocessed_text = ""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(dispatch_chat_completion, messages, preprocess_model) for messages in messages_list]
        for future in concurrent.futures.as_completed(futures):
            preprocessed_text += future.result()
    return preprocessed_text

parser = argparse.ArgumentParser()
parser.add_argument("--url", type=str, default=None, help="URL of PDF file")
parser.add_argument("--path", type=str, default=None, help="path of PDF file")
parser.add_argument("--keep-refs", action="store_true", help="keep references in filtered text")
args = parser.parse_args()

url = args.url
path = args.path
keep_refs = args.keep_refs

assert url != path, "Need to provide either a URL or a path"

config_file = open("config.json")
config = json.loads(config_file.read())
openai.api_key = config["OpenAI Key"]
config_file.close()

if url:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    path = '/tmp/paper.pdf'
    if response.headers['Content-Type'] == 'application/pdf':
        with open(path, 'wb') as f:
            f.write(response.content)
    else:
        print("The URL did not return a PDF file")
        exit()

reader = PdfReader(path)
raw_text = ""
for page in reader.pages:
    raw_text += page.extract_text()

preprocessed_text = preprocess_text(raw_text, keep_refs)

tokenizer = tiktoken.encoding_for_model("gpt-4")
num_tokens = len(tokenizer.encode(preprocessed_text))
print(f"Num tokens = {num_tokens}")

messages = [{"role": "user", "content": preprocessed_text}]

while True:
    request = input('> ')
    if request == "exit":
        break
    messages.append({"role": "user", "content": request})
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model = "gpt-4",
                temperature = 0,
                messages = messages
            )
            break
        except Exception as e:
            print(repr(e))
            time.sleep(random.uniform(5, 20))

    response = completion.choices[0].message
    messages.append(response)
    print(response["content"])

log_file = open("log.txt", "a")
now = datetime.datetime.now()
log_file.write("Conversation ended at " + now.strftime("%Y-%m-%d %H:%M:%S") + ":\n")
for message in messages:
    log_file.write(str(message) + "\n")
log_file.write("\n")
log_file.close()
