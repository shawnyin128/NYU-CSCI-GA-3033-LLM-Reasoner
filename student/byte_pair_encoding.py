import multiprocessing
import os
from collections import defaultdict

import regex as re
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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


def remove_special_tokens(corpus_chunk: str, special_tokens: list[str]):
    escaped_special_tokens = [re.escape(tok) for tok in special_tokens] # make sure special token including | can be
    special_tokens_str = "|".join(escaped_special_tokens) # | here stands for or
    clean_corpus_chunk = re.split(special_tokens_str, corpus_chunk) # split by any special token
    return clean_corpus_chunk


def bpe_pre_tokenize_worker(input_path: str, start: int, end: int, special_tokens: list[str]):
    word_freq = defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(start)
        # get the chunk text
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # split corpus based on special tokens
        chunk_wo_st = remove_special_tokens(chunk, special_tokens)
        # for each corpus, do pre-tokenization
        for text in chunk_wo_st:
            for match in re.finditer(PAT, text):
                token = match.group()
                byte_token = tuple(bytes([b]) for b in token.encode(
                    "utf-8"))  # bytes(b) means create an empty bytes with b length. bytes([b]) means create a bytes with b as the element
                word_freq[byte_token] += 1
    return word_freq


def get_pair(word_freq: dict[tuple[bytes], int]):
    pair_freq = defaultdict(int)
    for word, freq in word_freq.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freq[pair] += freq
    max_freq = -1
    best_pair = None
    for pair, freq in pair_freq.items():
        if freq > max_freq or (freq == max_freq and pair > best_pair):
            best_pair = pair
            max_freq = freq
    return best_pair


def update_word_freq(word_freq: dict[tuple[bytes], int], pair: tuple[bytes]):
    merged_symbol = pair[0] + pair[1]
    new_word_freq = defaultdict(int)

    for word, freq in word_freq.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged_symbol)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_freq[tuple(new_word)] += freq

    return new_word_freq


def train_bpe_naive(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab = {}
    for i in range(len(special_tokens)):
        vocab[i] = special_tokens[i].encode("utf-8")
    offset = len(special_tokens)
    for i in range(256):
        vocab[offset + i] = bytes([i])
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        # multiprocessing run
        tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        with multiprocessing.Pool(num_processes) as pool:
            word_freqs = pool.starmap(bpe_pre_tokenize_worker, tasks)

        # gather results
        word_freq = defaultdict(int)
        for local_dict in word_freqs:
            for word, freq in local_dict.items():
                word_freq[word] += freq

        # merge pair
        ## naive merge
        merge = []
        cur_idx = len(vocab)
        while cur_idx < vocab_size:
            pair = get_pair(word_freq)
            merge.append(pair)
            vocab[cur_idx] = pair[0] + pair[1]
            cur_idx += 1
            word_freq = update_word_freq(word_freq, pair)
    return vocab, merge







def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], mode: str = "fast"):
    if mode == "naive":
        return train_bpe_naive(input_path, vocab_size, special_tokens)
    else:
        pass


if __name__ == "__main__":
    train_bpe_naive("../data/TinyStoriesV2-GPT4-valid.txt", 32000, ["<|endoftext|>"])