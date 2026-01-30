
import regex as re
from collections import defaultdict
from multiprocessing import Pool
import os
import time
from typing import BinaryIO
from collections import Counter


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


# global in worker
TOKEN_PAT = None


def init_worker(token_pat):
    global TOKEN_PAT
    TOKEN_PAT = re.compile(token_pat)


def pretokenize_worker(content):
    counter = Counter()
    for m in TOKEN_PAT.finditer(content):
        tok = m.group()
        counter[tok.encode("utf-8")] += 1
    return counter


def process_chunk(args):
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pretokenize_worker(chunk)


def train_bpe(input_path, vocab_size, special_tokens, num_workers=None):

    if num_workers is None:
        num_workers = os.cpu_count()

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    escaped = [re.escape(tok) for tok in special_tokens]
    combined_pat = f"({'|'.join(escaped)})|({PAT})"
    compiled_pat = re.compile(combined_pat)

    # 获取文件大小决定是否使用多进程
    file_size = os.path.getsize(input_path)

    token_counter = Counter()

    # 小文件或单 worker 时使用单线程处理（避免分块导致的 token 切分问题）
    if num_workers <= 1 or file_size < 10_000_000:  # 小于 10MB 用单线程
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        for m in compiled_pat.finditer(content):
            tok = m.group()
            token_counter[tok.encode("utf-8")] += 1
    else:
        # 大文件使用多进程并行处理
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f,
                desired_num_chunks=num_workers,
                split_special_token=special_tokens[0].encode("utf-8") if special_tokens else b"",
            )
        tasks = [
            (input_path, s, e)
            for s, e in zip(boundaries[:-1], boundaries[1:])
        ]

        with Pool(
            num_workers,
            initializer=init_worker,
            initargs=(combined_pat,)
        ) as pool:
            results = pool.map(process_chunk, tasks)
            for c in results:
                token_counter.update(c)

    merges = []

    vocab = []
    vocab.extend(tok.encode("utf-8") for tok in special_tokens)

    for i in range(256):
        b = bytes([i])
        vocab.append(b)

    # 构建 special_tokens 的 bytes 集合，用于判断
    special_tokens_bytes = set(tok.encode("utf-8") for tok in special_tokens)

    word_freq = defaultdict(int)

    for word, freq in token_counter.items():
        # Bug 1 修复: special tokens 不拆分为字节，直接跳过不参与 BPE
        if word in special_tokens_bytes:
            continue  # special tokens 不参与 BPE 合并
        byte_tuple = tuple(bytes([b]) for b in word)
        word_freq[byte_tuple] += freq

    word_keys = list(word_freq.keys())

    idx_freq = defaultdict(int)
    for i in range(len(word_keys)):
        key = word_keys[i]
        idx_freq[i] = word_freq[key]

    pair_freq = defaultdict(int)
    source_info = defaultdict(set)

    for idx, value in idx_freq.items():
        key = word_keys[idx]
        # Bug 2 修复: 统计所有 pair 出现次数，而不是唯一 pair
        for i in range(len(key) - 1):
            pair = (key[i], key[i + 1])
            pair_freq[pair] += value
            source_info[pair].add(idx)

    while len(vocab) < vocab_size:
        if not pair_freq:
            break

        best_pair = max(pair_freq, key=lambda pair: (pair_freq[pair], pair))
        merges.append(best_pair)
        vocab.append(best_pair[0] + best_pair[1])

        affected_indices = list(source_info[best_pair])
        for idx in affected_indices:
            word = word_keys[idx]
            value = idx_freq[idx]

            new_word = []
            i = 0
            position = []
            new_position = []
            off_set = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(best_pair[0] + best_pair[1])
                    position.append(i)
                    new_position.append(i - off_set)
                    off_set += 1
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)

            word_keys[idx] = new_word

            # Bug 2 修复: 使用 Counter 统计 pair 出现次数的变化
            # 统计旧 word 中所有 pairs
            old_pair_count = Counter()
            for i in range(len(word) - 1):
                old_pair_count[(word[i], word[i + 1])] += 1

            # 统计新 word 中所有 pairs
            new_pair_count = Counter()
            for i in range(len(new_word) - 1):
                new_pair_count[(new_word[i], new_word[i + 1])] += 1

            # 计算差异并更新 pair_freq 和 source_info
            all_pairs = set(old_pair_count.keys()) | set(new_pair_count.keys())
            for pair in all_pairs:
                old_c = old_pair_count.get(pair, 0)
                new_c = new_pair_count.get(pair, 0)
                diff = new_c - old_c

                if diff != 0:
                    pair_freq[pair] += diff * value

                    if pair_freq[pair] <= 0:
                        del pair_freq[pair]
                        if pair in source_info:
                            del source_info[pair]
                    else:
                        # 更新 source_info
                        if new_c > 0 and old_c == 0:
                            source_info[pair].add(idx)
                        elif new_c == 0 and old_c > 0:
                            source_info[pair].discard(idx)

    vocab_dict = {i: vocab[i] for i in range(len(vocab))}
    return vocab_dict, merges


if __name__ == "__main__":
    input_path = '../data/TinyStories-valid.txt'
    special_tokens = ["<|endoftext|>"]

    s = time.time()
    vocab, merges = train_bpe(input_path, 1000, special_tokens)
    e = time.time()
    print('use time', e - s)
    print(vocab[:30], merges[:30])
