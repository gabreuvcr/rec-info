import re
import os
import json
import nltk
import time
import unicodedata
from queue import Queue
from threading import Thread
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from merge_indexes import merge_partial_indexes

MEGABYTE = 1024 * 1024
SENTINEL = object()

NUM_THREADS = 3
LINE_LIMIT = 10_000
DOC_LIMIT = 1_000

# NUM_THREADS = 2
# LINE_LIMIT = 100
# DOC_LIMIT = 50

DOC_QUEUE = Queue(maxsize=5_000)
STOP_WORDS = set(stopwords.words('english'))
SNOWBALL_STEMMER = SnowballStemmer(language='english')
# process = psutil.Process(os.getpid())


def download_resources(quiet: bool = True):
    nltk.download('punkt', quiet=quiet)
    nltk.download("stopwords", quiet=quiet)


def parse_document(document: dict) -> str:
    parsed_document = document['title'] + ' ' + document['text']
    for keyword in document['keywords']:
        parsed_document += ' ' + keyword
    return parsed_document


def remove_accents(document: str):
    return ''.join(
        char for char in unicodedata.normalize('NFD', document)
        if unicodedata.category(char) != 'Mn'
    )


def clean_text(document: str) -> str:
    lower_document = document.casefold()
    document_without_accent = remove_accents(lower_document)
    cleaned_document = re.sub(r'[^0-9a-zA-Z]+', ' ', document_without_accent)
    return cleaned_document


def split_alphanumeric(token):
    regex = r'(\d+|[a-zA-Z]+)'
    return re.findall(regex, token)


def produce_tokens(document: str) -> list[tuple[str, int]]:
    cleaned_document = clean_text(document)
    tokens = word_tokenize(cleaned_document)
    
    # Handling numbers
    # tokens = [token for token in tokens if token.isalpha()]
    tokens = [t for tokens in tokens for t in split_alphanumeric(tokens)]
    tokens = [str(int(token)) if token.isnumeric() else token for token in tokens]

    # Removing stop words
    tokens = [token for token in tokens if not token in STOP_WORDS]

    # Stemming the words
    # tokens = [SNOWBALL_STEMMER.stem(token) for token in tokens]

    # Counting the frequency of each token
    tokens_frequency = {}
    for token in tokens:
        tokens_frequency[token] = tokens_frequency.get(token, 0) + 1
  
    tokens = [(token, freq) for token, freq in tokens_frequency.items()]
    return tokens


def tokenize(document: dict) -> list[tuple[str, int]]:
    parsed_document = parse_document(document)
    tokens = produce_tokens(parsed_document)
    return tokens
   

def index(document: dict, inverted_index: dict) -> None:
    doc_id = int(document['id'])
    for (token, freq) in tokenize(document):
        if token not in inverted_index:
            inverted_index[token] = []
        inverted_index[token] += [(doc_id, freq)]


def reader(corpus_path: str) -> None:
    line_count = 0
    with open(corpus_path, 'r') as corpus_file:
        for line in corpus_file:
            if line_count == LINE_LIMIT: break
            document = json.loads(line)
            DOC_QUEUE.put(document)
            line_count += 1
    for _ in range(NUM_THREADS):
        DOC_QUEUE.put(SENTINEL)
    print('end reader')


def write_to_file(
        index_dir_path: str, 
        inverted_index: dict, 
        consumer_id: int, 
        doc_count: int
    ) -> None:
    if not os.path.exists(f"{index_dir_path}/tmp"):
        os.makedirs(f"{index_dir_path}/tmp")
    with open(f"{index_dir_path}/tmp/index_{consumer_id}_{doc_count}.txt", 'w') as index_file:
        for token in sorted(inverted_index):
            index_file.write(f'{token}|')
            for i in range(len(inverted_index[token])):
                doc_id, freq = inverted_index[token][i]
                if i != len(inverted_index[token]) - 1:
                    index_file.write(f'{doc_id},{freq}|')
                else:
                    index_file.write(f'{doc_id},{freq}')
            index_file.write('\n')
    inverted_index.clear()


def consumer(index_dir_path: str, id: int) -> None:
    inverted_index = {}
    doc_count = 0
    for document in iter(DOC_QUEUE.get, SENTINEL):
        index(document, inverted_index)
        doc_count += 1
        if doc_count % DOC_LIMIT == 0:
            write_to_file(index_dir_path, inverted_index, id, doc_count)
    if inverted_index:
        write_to_file(index_dir_path, inverted_index, id, doc_count)
    print(f'end consumer {id}')


def produce_partial_indexes(corpus_path: str, index_dir_path: str) -> None:
    if not os.path.exists(index_dir_path):
        os.makedirs(index_dir_path)
    
    download_resources()
    
    start_time = time.time()
    
    reader_thread = Thread(
        target=reader, 
        args=(corpus_path,)
    )
    reader_thread.start()
    consumer_threads = [
        Thread(target=consumer, args=(index_dir_path, i + 1))
        for i in range(NUM_THREADS)
    ]
    for consumer_thread in consumer_threads:
        consumer_thread.start()
    
    reader_thread.join()
    for consumer_thread in consumer_threads:
        consumer_thread.join()

    end_time = time.time()

    print(f'Index time: {end_time - start_time:.2f} seconds')


def produce_index(corpus_path: str, index_dir_path: str) -> None:
    if index_dir_path[-1] == '/': index_dir_path = index_dir_path[:-1]
    
    start_time = time.time()
    
    produce_partial_indexes(corpus_path, index_dir_path)
    merge_partial_indexes(index_dir_path)
    
    end_time = time.time()
    print(f'Elapsed time: {end_time - start_time:.2f} seconds')
