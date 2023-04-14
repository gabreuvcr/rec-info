import re
import os
import gc
import sys
import json
import nltk
import time
import psutil
import resource
import argparse
from queue import Queue
from threading import Thread, Lock
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
# from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

MEGABYTE = 1024 * 1024
SENTINEL = object()

num_threads = 2
inverted_index = {}
write_id = 1
doc_count = 0
lock = Lock()
doc_queue = Queue(maxsize=1_000)
stop_words = set(stopwords.words('english'))
snowball_stemmer = SnowballStemmer(language='english')
regexp_tokenizer = RegexpTokenizer(r'\d+\.\d+|[a-zA-Z0-9]+')
process = psutil.Process(os.getpid())

def memory_limit(value: int):
    limit = value * MEGABYTE
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def download_resources(quiet: bool = True):
    nltk.download('punkt', quiet=quiet)
    nltk.download("stopwords", quiet=quiet)


def merge_document_fields(document: dict) -> str:
    merged_document = document['title'] + ' ' + document['text']
    for keyword in document['keywords']:
        merged_document += ' ' + keyword
    return merged_document


def produce_tokens(document: str) -> list[str]:
    document = document.lower()
    # Removing all characters except alphabets, numbers, dot, hyphen and underscore
    clean_text_regex = r'[^A-Za-z0-9\.\-\_ ]+'
    document = re.sub(clean_text_regex, '', document)
    # Tokenizing the document per alphanumeric and decimals
    tokens = regexp_tokenizer.tokenize(document)
    # Removing stop words and stemming the words
    tokens = [w for w in tokens if not w in stop_words]
    # tokens = [snowball_stemmer.stem(w) for w in tokens]
    return tokens


def produce_token_frequency(raw_tokens: list[str]) -> list[tuple[str, int]]:
    raw_tokens = sorted(raw_tokens)
    i, j, tokens = 0, 0, []
    while i < len(raw_tokens):
        if raw_tokens[i] != raw_tokens[j]:
            tokens += [(raw_tokens[j], i - j)]
            j = i
        i += 1
    if j < len(raw_tokens):
        tokens += [(raw_tokens[j], i - j)]
    return tokens


def tokenize(document: dict) -> list[tuple[str, int]]:
    merged_document = merge_document_fields(document)
    raw_tokens = produce_tokens(merged_document)
    tokens = produce_token_frequency(raw_tokens)
    return tokens
   

def index(document: dict) -> None:
    d_id = int(document['id'])
    for (term, freq) in tokenize(document):
        # with lock:
        if term not in inverted_index:
            inverted_index[term] = []
        inverted_index[term] += [(d_id, freq)]


def reader(corpus_path: str) -> None:
    line_count = 0
    with open(corpus_path, 'r') as corpus_file:
        for line in corpus_file:
            if line_count == 10_000: break
            document = json.loads(line)
            doc_queue.put(document)
            line_count += 1
    for _ in range(num_threads):
        doc_queue.put(SENTINEL)
    print('end reader')


def write_to_file(index_dir_path: str) -> None:
    global write_id
    with open(f"{index_dir_path}/index_{write_id}.txt", 'w') as index_file:
        index_file.write(f'{inverted_index}')
    write_id += 1
    inverted_index.clear()


def consumer(index_dir_path: str, id: int) -> None:
    global doc_count
    for document in iter(doc_queue.get, SENTINEL):
        with lock:
            index(document)
            doc_count += 1
            if doc_count == 10_000:
                write_to_file(index_dir_path)
                doc_count = 0
    with lock:
        if inverted_index:
            # print(inverted_index)
            write_to_file(index_dir_path)
            doc_count = 0
    print(f'end consumer {id}')


def main(corpus_path: str, index_dir_path: str, memory_limit: int) -> None:
    memory_limit = memory_limit * MEGABYTE
    if not os.path.exists(index_dir_path):
        os.makedirs(index_dir_path)
    
    download_resources()

    start_time = time.time()

    reader_thread = Thread(
        target=reader, 
        args=(corpus_path,)
    )
    reader_thread.start()
    consumers_thread: list[Thread] = []
    for t in range(num_threads):
        consumers_thread += [
            Thread(
                target=consumer, 
                args=(index_dir_path, t + 1)
            )
        ]
        consumers_thread[t].start()

    reader_thread.join()
    for t in range(num_threads):
        consumers_thread[t].join()

    end_time = time.time()
    print(f'{end_time - start_time:.4f} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '-m',
        dest='memory_limit',
        action='store',
        required=True,
        type=int,
        help='memory available'
    )
    parser.add_argument(
        '-c',
        dest='corpus_path',
        action='store',
        required=True,
        type=str,
        help='corpus file'
    )
    parser.add_argument(
        '-i',
        dest='index_dir_path',
        action='store',
        required=True,
        type=str,
        help='index directory'
    )
    args = parser.parse_args()
    memory_limit(args.memory_limit)
    try:
        main(args.corpus_path, args.index_dir_path, args.memory_limit)
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)
    except KeyboardInterrupt:
        print(" interrupting...")
        sys.exit(1)
