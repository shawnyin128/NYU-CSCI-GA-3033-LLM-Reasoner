import json
import multiprocessing
import os
from collections import defaultdict

import regex as re
from typing import BinaryIO, Iterable

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


def bpe_pre_tokenize(corpus_chunk: str, special_tokens: list[str]):
    byte_tokens = []
    chunk_wo_st = remove_special_tokens(corpus_chunk, special_tokens)
    for text in chunk_wo_st:
        for match in re.finditer(PAT, text):
            token = match.group()
            byte_token = tuple(bytes([b]) for b in token.encode("utf-8"))
            byte_tokens.append(byte_token)
    return byte_tokens


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


def update_word_and_pair_freq(word_freq: dict[tuple[bytes], int], pair: tuple[bytes], pair_freq: dict[tuple[bytes], int]):
    merged_symbol = pair[0] + pair[1]
    new_word_freq = defaultdict(int)

    for word, freq in word_freq.items():
        if pair not in zip(word, word[1:]):
            new_word_freq[word] += freq
            continue

        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                if i > 0:
                    pair_freq[(word[i - 1], word[i])] -= freq
                if i < len(word) - 2:
                    if (word[i + 1], word[i + 2]) != pair:
                        pair_freq[(word[i + 1], word[i + 2])] -= freq
                new_word.append(merged_symbol)
                if i > 0:
                    pair_freq[(word[i - 1], merged_symbol)] += freq
                if i < len(word) - 2:
                    if (word[i + 1], word[i + 2]) != pair:
                        pair_freq[(merged_symbol, word[i + 2])] += freq
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
        merges = []
        cur_idx = len(vocab)
        while cur_idx < vocab_size:
            pair = get_pair(word_freq)
            merges.append(pair)
            vocab[cur_idx] = pair[0] + pair[1]
            cur_idx += 1
            word_freq = update_word_freq(word_freq, pair)
    return vocab, merges


def train_bpe_fast(input_path: str, vocab_size: int, special_tokens: list[str]):
    # steps:
    # 1. init vocab: init dict, insert special tokens, insert 256 tokens
    # 2. open file
    # 3. split chunk
    # 4. for each chunk, do multiprocessing to pre-tokenize. merge to get global frequency
    # 5. create pair frequency dict
    # 6. do merging

    # 1. init vocab
    vocab = {}
    for i in range(len(special_tokens)):
        vocab[i] = special_tokens[i].encode("utf-8")
    offset = len(special_tokens)
    for i in range(256):
        vocab[offset + i] = bytes([i])

    # 2. open file
    with open(input_path, "rb") as f:
        # 3. split chunk
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # 4. multiprocessing: pretokenize
        tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
        with multiprocessing.Pool(num_processes) as pool:
            word_freqs = pool.starmap(bpe_pre_tokenize_worker, tasks)
        word_freq = defaultdict(int)
        for local_dict in word_freqs:
            for old_word, freq in local_dict.items():
                word_freq[old_word] += freq

        # 5. global pair frequency dict
        pair_freq = defaultdict(int)
        pair_words = defaultdict(set)

        for old_word, freq in word_freq.items():
            for i in range(len(old_word) - 1):
                pair = (old_word[i], old_word[i + 1])
                pair_freq[pair] += freq
                pair_words[pair].add(old_word)

        # 6. do merging
        merges = []
        cur_idx = len(vocab)
        while cur_idx < vocab_size:
            # 6.1. find current best pair
            best_pair = None
            max_freq = -1
            for pair, freq in pair_freq.items():
                if freq > max_freq or (freq == max_freq and pair > best_pair):
                    best_pair = pair
                    max_freq = freq

            # 6.2. append the best merge
            merges.append(best_pair)
            merge_token = best_pair[0] + best_pair[1]
            vocab[cur_idx] = merge_token
            cur_idx += 1

            # 6.3. find affected words
            affected_words = pair_words[best_pair]
            del pair_words[best_pair]
            del pair_freq[best_pair]

            # 6.4. update frequency
            for old_word in affected_words:
                freq = word_freq[old_word]

                # for all pairs in this word, its pair frequency should be updated, like this old pairing is not valid
                for i in range(len(old_word) - 1):
                    old_pair = (old_word[i], old_word[i + 1])
                    # remove the affected pair frequency
                    pair_freq[old_pair] -= freq
                    # remove the affected word pair map
                    pair_words[old_pair].discard(old_word)

                # merge
                new_word = [] # contains the new byte pairs
                i = 0
                while i < len(old_word):
                    if i < len(old_word) - 1 and (old_word[i], old_word[i + 1]) == best_pair:
                        new_word.append(merge_token)
                        i += 2
                    else:
                        new_word.append(old_word[i])
                        i += 1
                new_word = tuple(new_word)

                # pop up the current word pair from word_freq like it never existed before.
                # add new word pair to word_freq
                word_freq.pop(old_word)
                word_freq[new_word] += freq

                # update pair_freq and pair_words
                for i in range(len(new_word) - 1):
                    pair = (new_word[i], new_word[i + 1])
                    pair_freq[pair] += freq
                    pair_words[pair].add(new_word)
    return vocab, merges


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], mode: str = "fast"):
    if mode == "naive":
        return train_bpe_naive(input_path, vocab_size, special_tokens)
    else:
        return train_bpe_fast(input_path, vocab_size, special_tokens)


def save_bpe(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_path: str, type: str = "train"):
    serializable_vocab = {
        str(k): v.decode("latin-1")
        for k, v in vocab.items()
    }

    with open(os.path.join(output_path, f"vocab_{type}.json"), "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False)

    serializable_merges = [
        (a.decode("latin-1"), b.decode("latin-1"))
        for a, b in merges
    ]

    with open(os.path.join(output_path, f"merges_{type}.json"), "w", encoding="utf-8") as f:
        json.dump(serializable_merges, f, ensure_ascii=False)


def load_bpe(vocab_filepath: str, merges_filepath: str, type: str = "train"):
    vocab_path = os.path.join(vocab_filepath, f"vocab_{type}.json")
    merges_path = os.path.join(merges_filepath, f"merges_{type}.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

        vocab = {
            int(k): v.encode("latin-1")
            for k, v in vocab.items()
        }

    with open(merges_path, "r", encoding="utf-8") as f:
        merges = json.load(f)

        merges = [
            (a.encode("latin-1"), b.encode("latin-1"))
            for a, b in merges
        ]
    return vocab, merges


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def _preprocess_vocab_and_merges(self):
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.merges_map = {merge: i for i, merge in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        vocab, merges = load_bpe(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str):
        # 1. pre-tokenize, represent using UTF-8 bytes
        pre_tokenized_text = bpe_pre_tokenize(text, self.special_tokens)
        # 2. apply merges
        for i in range(len(pre_tokenized_text)):
            word = pre_tokenized_text[i]
            while True:
                best_pair = None
                best_index = None
                best_rank = float("inf")
                for j in range(len(word) - 1):
                    pair = (word[j], word[j + 1])
                    if pair in self.merges_map:
                        rank = self.merges_map[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                            best_index = j
                if best_pair is not None:
                    merge_token = best_pair[0] + best_pair[1]
                    word = word[:best_index] + (merge_token,) + word[best_index + 2:]
                else:
                    break
            pre_tokenized_text[i] = word
        # 3. go to vocab and get ids
        ids = []
        for word in pre_tokenized_text:
            for tok in word:
                ids.append(self.inv_vocab[tok])
        return ids

    def encode_iterable(self, iterable: Iterable[str]):
        for text in iterable:
            ids = self.encode(text)
            for id in ids:
                yield id

    def decode(self, ids: list[int]):
        pass

if __name__ == "__main__":
    # import time
    # import tracemalloc
    # import cProfile
    # import pstats
    #
    # start_time = time.time()
    #
    # vocab, merge = train_bpe("../data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"], "fast")
    #
    # end_time = time.time()
    #
    # print("Time:", end_time - start_time)
    #
    # save_bpe(vocab, merge, "./checkpoint/BPE", "train")
#
#     cProfile.run(
#         'train_bpe_naive("../data/TinyStoriesV2-GPT4-valid.txt", 10000, ["<|endoftext|>"])',
#         'profile_output'
#     )
#     stats = pstats.Stats('profile_output')
#     stats.sort_stats('cumulative').print_stats(20)
    text = 'the cat ate'
    print(bpe_pre_tokenize(text, ["<|endoftext|>"]))