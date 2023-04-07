import re
import sys
import json
import nltk
import time
import resource
import argparse
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

MEGABYTE = 1024 * 1024

def memory_limit(value: int):
    limit = value * MEGABYTE
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def merge_document_fields(document: dict) -> str:
    merged_document = document['title'] + ' ' + document['text']
    for keyword in document['keywords']:
        merged_document += ' ' + keyword
    return merged_document


def produce_tokens(document: str) -> list[str]:
    nltk.download('punkt', quiet=True)
    nltk.download("stopwords", quiet=True)
    document = document.lower()
    
    #Removing all characters except alphabets, numbers,
    # dot, hyphen and underscore
    document = re.sub(r'[^A-Za-z0-9\.\-\_ ]+', '', document)
    
    # Tokenizing the document per alphanumeric and decimals
    tokenizer = RegexpTokenizer(r'\d+\.\d+|[a-zA-Z0-9]+')
    tokens = tokenizer.tokenize(document)

    # Removing stop words and stemming the words
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return tokens


def produce_token_frequency(raw_tokens: list[str]) -> list[tuple[str, int]]:
    raw_tokens = sorted(raw_tokens)
    i, j, tokens = 0, 0, []
    while i < len(raw_tokens):
        if raw_tokens[i] != raw_tokens[j]:
            tokens += [(raw_tokens[j], i - j)]
            j = i
        i += 1
    tokens += [(raw_tokens[j], i - j)]
    return tokens


def tokenize(document: dict) -> list[tuple[str, int]]:
    print(document, end="\n\n")
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


def main(corpus_path: str, index_dir_path: str) -> None:
    start_time = time.time()
    with open(corpus_path, 'r') as f:
        corpus = [json.loads(line) for line in f]
    
    inverted_index = index(corpus[:2])
    print(inverted_index)
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
        main(args.corpus_path, args.index_dir_path)
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)
