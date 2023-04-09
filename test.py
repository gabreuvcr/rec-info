import os
import sys
import time
import psutil
import argparse
import resource
import threading

MEGABYTE = 1024 * 1024

def memory_limit(value: int):
    limit = value * MEGABYTE
    print(limit)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

def main(limit) -> None:
    l = []
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)
    for i in range(25_500_000):
        l += [i]
            # if process.memory_info().rss > limit - 50 * MEGABYTE:
        # print(f'oi {i}')
        if i % 1000 == 0 and process.memory_info().rss / limit > 0.50:
            print(f'Memory limit reached {i}')
            print(process.memory_info().rss)
            l.clear()
            print(process.memory_info().rss)
            # break
        # if percent > 90:
            # print(f'Memory limit reached {i}')
            # break
    print(process.memory_info().rss)
    print(process.memory_info().rss / limit)

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
    args = parser.parse_args()
    memory_limit(args.memory_limit)
    try:
        main(args.memory_limit * MEGABYTE)
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)
