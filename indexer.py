import sys
import re
import resource
import argparse
import json
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords

MEGABYTE = 1024 * 1024

def memory_limit(value: int):
    limit = value * MEGABYTE
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def merge_document_fields(document: dict) -> str:
    merged_document = document['title'] + ' ' + document['text']
    for keyword in document['keywords']:
        merged_document += ' ' + keyword
    return merged_document


def remove_dot_between_words(document: str) -> str:
    new_document = ''
    for i in range(len(document)):
        if document[i] == '.':
            if (
                i > 0 
                and i < len(document) - 1
                and document[i - 1].isdigit()
                and document[i + 1].isdigit()
            ):
                new_document += document[i]
            else:
                new_document += ' '
        else:
            new_document += document[i]
    return new_document


def clean_document(document: str) -> str:
    document = document.casefold()
    document = document.replace('-', ' ')
    document = remove_dot_between_words(document)
    document = re.sub('[^A-Za-z0-9. ]+', '', document)

    return document


def produce_tokens(document: str) -> list[str]:
    # nltk.download('punkt')
    # nltk.download("stopwords")
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(document)

    # twograms =  list(ngrams(tokens, 2))
    # tokens += [' '.join(twogram) for twogram in twograms]
    
    tokens = [w for w in tokens if not w in stop_words and len(w) >= 2]
    tokens = [stemmer.stem(w) for w in tokens]

    return tokens


def produce_token_frequency(raw_tokens: list[str]) -> list[tuple[str, int]]:
    raw_tokens = sorted(raw_tokens)
    
    tokens = []
    i, j = 0, 0
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
    cleared_document = clean_document(merged_document)
    raw_tokens = produce_tokens(cleared_document) 
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
    with open(corpus_path, 'r') as f:
        corpus = [json.loads(line) for line in f]
    
    inverted_index = index(corpus[:1])
    print(inverted_index)


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
