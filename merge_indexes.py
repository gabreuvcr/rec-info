import os
import time
import shutil

def merge_two_indexes(path_index_a, path_index_b, path_merged_index):
    def parse_torken_postings(line):
        posting = line.split('|')
        token = posting[0]
        postings = posting[1:]
        return token, postings

    with (
        open(path_index_a, 'r') as a,
        open(path_index_b, 'r') as b,
        open(path_merged_index, 'w') as c
    ):
        line_a, line_b = a.readline(), b.readline()
        while line_a and line_b:
            token_a, postings_a = parse_torken_postings(line_a)
            token_b, postings_b = parse_torken_postings(line_b)
            min_token = min(token_a, token_b)
            postings = []
            if token_a == token_b:
                postings = sorted(
                    postings_a + postings_b, 
                    key=lambda p: int(p.split(',')[0])
                )
                line_a, line_b = a.readline(), b.readline()
            elif token_a == min_token:
                postings = postings_a
                line_a = a.readline()
            elif token_b == min_token:
                postings = postings_b
                line_b = b.readline()
            
            c.write(f'{min_token}|')
            for i, posting in enumerate(postings):
                if posting[-1] == '\n': posting = posting[:-1]
                if i < len(postings) - 1:
                    c.write(f'{posting}|')
                else:
                    c.write(f'{posting}\n')
        while line_a:
            c.write(line_a)
            line_a = a.readline()
        while line_b:
            c.write(line_b)
            line_b = b.readline()


def merge_partial_indexes(index_dir_path: str) -> None:
    start_time = time.time()
    index_files = os.listdir(f'{index_dir_path}/tmp')
    merged_file = ''
    if len(index_files) >= 2:
        for i in range(len(index_files) - 1):
            index_a = index_files[i]
            index_b = index_files[i + 1]
            merged_file = f'merged_{i + 1}.txt'
            merge_two_indexes(
                f'{index_dir_path}/tmp/{index_a}',
                f'{index_dir_path}/tmp/{index_b}',
                f'{index_dir_path}/tmp/{merged_file}',
            )
            index_files[i + 1] = merged_file
    else:
        merged_file = index_files[0]
    
    os.rename(f'{index_dir_path}/tmp/{merged_file}', f'{index_dir_path}/inverted_index.txt')
    shutil.rmtree(f'{index_dir_path}/tmp')
    end_time = time.time()
    print(f'Merge time: {end_time - start_time:.2f} seconds')
