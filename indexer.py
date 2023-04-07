import sys
import re
import resource
import argparse
import json
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

MEGABYTE = 1024 * 1024
def memory_limit(value: int):
    limit = value * MEGABYTE
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

def tokenize(text: str) -> list[tuple[str, int]]:
    def remove_dot_between_words(text: str) -> str:
        new_text = ''
        for i in range(len(text)):
            if text[i] == '.':
                if (
                    i > 0 
                    and i < len(text) - 1
                    and text[i - 1].isdigit()
                    and text[i + 1].isdigit()
                ):
                    new_text += text[i]
                else:
                    new_text += ' '
            else:
                new_text += text[i]
        return new_text

    # nltk.download('punkt')
    # nltk.download("stopwords")
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    print(text, end="\n\n")
    text = text.casefold()
    text = text.replace('-', ' ')
    text = remove_dot_between_words(text)
    text = re.sub('[^A-Za-z0-9. ]+', '', text)

    raw_tokens = word_tokenize(text)
    raw_tokens = [w for w in raw_tokens if not w in stop_words and len(w) >= 2]
    raw_tokens = [stemmer.stem(w) for w in raw_tokens]
    
    tokens = []
    uniques_words = set(raw_tokens)
    for word in uniques_words:
        tokens += [(word, raw_tokens.count(word))]

    return tokens

def index(corpus: str) -> dict[str, list[tuple[int, int]]]:
    inverted_index = {}
    d_id = 0
    for document in corpus:
        document = merge_corpus_fields(document)
        d_id += 1
        for (term, freq) in tokenize(document):
            if term not in inverted_index:
                inverted_index[term] = []
            inverted_index[term] += [(d_id, freq)]
    return inverted_index

def merge_corpus_fields(line: dict) -> str:
    merged_text = line['title'] + ' ' + line['text']
    for keyword in line['keywords']:
        merged_text += ' ' + keyword
    return merged_text

def main(corpus_path: str, index_dir_path: str) -> None:
    with open(corpus_path, 'r') as f:
        corpus = [json.loads(line) for line in f]
    
    # inverted_index = index(corpus)
    # inverted_index = index(corpus[8:9])
    inverted_index = index(corpus[:18])
    # inverted_index = index(corpus[:4])
    print()
    print(dict(sorted(inverted_index.items(), key=lambda item: item[0])))


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
