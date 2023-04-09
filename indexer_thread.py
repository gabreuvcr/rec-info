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
import threading
from queue import Queue
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

nltk.download('punkt', quiet=True)
nltk.download("stopwords", quiet=True)

NUM_THREADS = 2
DOC_COUNT = Queue()
SENTINEL = object()
MEGABYTE = 1024 * 1024
INVERTED_INDEX = defaultdict(list)

i_id = 1
index_lock = threading.Lock()
process = psutil.Process(os.getpid())
stop_words = set(stopwords.words('english'))


def memory_limit(value: int):
    limit = value * MEGABYTE
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def merge_document_fields(document: dict) -> str:
    merged_document = document['title'] + ' ' + document['text']
    for keyword in document['keywords']:
        merged_document += ' ' + keyword
    return merged_document


def produce_tokens(document: str) -> list[str]:
    # snowball_stemmer = SnowballStemmer(language='english')
    text_clean_regex = r'[^A-Za-z0-9\.\-\_ ]+'
    regexp_tokenizer = RegexpTokenizer(r'\d+\.\d+|[a-zA-Z0-9]+')

    document = document.lower()

    # #Removing all characters except alphabets, numbers, dot, hyphen and underscore
    document = re.sub(text_clean_regex, '', document)
    
    # # Tokenizing the document per alphanumeric and decimals
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
   

def index(document: dict, inverted_index: dict) -> None:
    d_id = int(document['id'])
    for (term, freq) in tokenize(document):
        # if term not in inverted_index:
        #     inverted_index[term] = []
        # inverted_index[term] += [(d_id, freq)]
        INVERTED_INDEX[term].append((d_id, freq))


def reader(corpus_path: str, inqueue: Queue) -> None:
    line_count = 0
    with open(corpus_path, 'r') as corpus_file:
        for line in corpus_file:
            line_count += 1
            if line_count == 90_000: break
            inqueue.put(line)
    for _ in range(NUM_THREADS):
        inqueue.put(SENTINEL)
    print('end reader')

def write_to_file(index_dir_path: str, consumer_id: int, inverted_index: dict) -> None:
    import datetime
    global i_id
    print(f"liberando {consumer_id} {datetime.datetime.now()}")
    with open(f"{index_dir_path}/index_{i_id}.out", 'w') as index_file:
        index_file.write(f'{INVERTED_INDEX}')
        index_file.close()
        i_id += 1
    INVERTED_INDEX.clear()
    # gc.collect()


def consumer(inqueue: Queue, memory_limit: int, index_dir_path: str, id: int) -> None:
    global INVERTED_INDEX
    process = psutil.Process(os.getpid())
    inverted_index = {}
    for line in iter(inqueue.get, SENTINEL):
        document = json.loads(line)
        index(document, inverted_index)
        with index_lock:
            DOC_COUNT.put(True)
            print(DOC_COUNT.qsize(), process.memory_info().rss / MEGABYTE)
            if DOC_COUNT.qsize() >= 2_000:
                write_to_file(index_dir_path, id, inverted_index)
                DOC_COUNT.queue.clear()
    with index_lock:
        if INVERTED_INDEX:
            write_to_file(index_dir_path, id, inverted_index)
    DOC_COUNT.queue.clear()
    print(f'end consumer {id}')


def main(corpus_path: str, index_dir_path: str, memory_limit: int) -> None:
    memory_limit = memory_limit * MEGABYTE
    if not os.path.exists(index_dir_path):
        os.makedirs(index_dir_path)
    
    inqueue = Queue(maxsize=50)
    
    start_time = time.time()

    reader_thread = threading.Thread(target=reader, args=(corpus_path, inqueue))
    reader_thread.start()
    consumers: list[threading.Thread] = []
    for i in range(NUM_THREADS):
        c = threading.Thread(target=consumer, args=(inqueue, memory_limit, index_dir_path, i + 1))
        consumers += [c]
        c.start()

    reader_thread.join()
    for c in consumers:
        c.join()

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
