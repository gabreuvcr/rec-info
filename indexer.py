import re
import os
import sys
import json
import nltk
import time
import resource
import argparse
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

nltk.download('punkt', quiet=True)
nltk.download("stopwords", quiet=True)

MEGABYTE = 1024 * 1024
STOPWORDS = set(stopwords.words('english'))
STEMMER = SnowballStemmer(language='english')
TEXT_CLEAN_REGEX = r'[^A-Za-z0-9\.\-\_ ]+'
TOKENIZER = RegexpTokenizer(r'\d+\.\d+|[a-zA-Z0-9]+')

def memory_limit(value: int):
    limit = value * MEGABYTE
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def merge_document_fields(document: dict) -> str:
    merged_document = document['title'] + ' ' + document['text']
    for keyword in document['keywords']:
        merged_document += ' ' + keyword
    return merged_document


def produce_tokens(document: str) -> list[str]:
    document = document.lower()
    
    #Removing all characters except alphabets, numbers, dot, hyphen and underscore
    document = re.sub(TEXT_CLEAN_REGEX, '', document)
    
    # Tokenizing the document per alphanumeric and decimals
    tokens = TOKENIZER.tokenize(document)

    # Removing stop words and stemming the words
    tokens = [w for w in tokens if not w in STOPWORDS]
    tokens = [STEMMER.stem(w) for w in tokens]
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
   

def index(corpus: list[dict]) -> dict[str, list[tuple[int, int]]]:
    inverted_index = {}
    d_id = 0
    for document in corpus:
        d_id += 1
        for (term, freq) in tokenize(document):
            if term not in inverted_index:
                inverted_index[term] = []
            inverted_index[term] += [(d_id, freq)]
    return inverted_index


def main(corpus_path: str, index_dir_path: str, memory_limit: int) -> None:
    if not os.path.exists(index_dir_path):
        os.makedirs(index_dir_path)
    
    start_time = time.time()
    with open(corpus_path, 'r') as corpus_file:
        i_id = 1
        while True:
            corpus_block = corpus_file.readlines(MEGABYTE * 10)

            if not corpus_block: break
            
            corpus = [json.loads(document) for document in corpus_block]

            inverted_index = index(corpus)

            index_file = open(f"{index_dir_path}/index_{i_id}.out", 'w')
            index_file.write(f'{inverted_index}')
            index_file.close()

            corpus.clear()
            i_id += 1
            break

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
