import json
import multiprocessing
import os
import time
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


def remove_special_tokens(corpus_chunk: str, special_tokens: list[str], keep_special_token: bool = False):
    clean_corpus_chunk = corpus_chunk
    if special_tokens is not None:
        if keep_special_token:
            special_tokens_str = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
        else:
            special_tokens_str = "|".join([re.escape(tok) for tok in special_tokens])

        clean_corpus_chunk = re.split(special_tokens_str, corpus_chunk) # split by any special token
    return clean_corpus_chunk


def bpe_pre_tokenize_worker(input_path: str, start: int, end: int, special_tokens: list[str]):
    word_freq = defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(start)
        # get the chunk text
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # split corpus based on special tokens
        chunk_wo_st = remove_special_tokens(chunk, special_tokens, keep_special_token=False)
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

        self._preprocess_vocab_and_merges()

    def _preprocess_vocab_and_merges(self):
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.merges_map = {merge: i for i, merge in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        vocab, merges = load_bpe(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def _bpe_pre_tokenize(self, corpus_chunk: str):
        # split corpus based on special tokens but do not remove special tokens
        if self.special_tokens is not None:
            segment = remove_special_tokens(corpus_chunk, self.special_tokens, keep_special_token=True)
        else:
            segment = [corpus_chunk]

        # for each segment, do pre-tokenization
        byte_tokens = []
        for seg in segment:
            if self.special_tokens is not None and seg in self.special_tokens:
                byte_tokens.append((seg.encode("utf-8"),))
            else:
                for match in re.finditer(PAT, seg):
                    token = match.group()
                    byte_token = tuple(bytes([b]) for b in token.encode("utf-8"))
                    byte_tokens.append(byte_token)
        return byte_tokens

    def encode(self, text: str):
        # 1. pre-tokenize, represent using UTF-8 bytes
        pre_tokenized_text = self._bpe_pre_tokenize(text)
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
        byte_candidates = []
        for id in ids:
            byte_candidates.append(self.vocab[id])
        full_byte = b"".join(byte_candidates)
        return full_byte.decode("utf-8", errors="replace")


if __name__ == "__main__":
    # import time
    # import tracemalloc
    # start_time = time.time()
    # vocab, merge = train_bpe("../data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"], "fast")
    # end_time = time.time()
    # print("Time:", end_time - start_time)
    # save_bpe(vocab, merge, "./checkpoint/BPE", "train")

    # import cProfile
    # import pstats
    # cProfile.run(
    #     'train_bpe_naive("../data/TinyStoriesV2-GPT4-valid.txt", 10000, ["<|endoftext|>"])',
    #     'profile_output'
    # )
    # stats = pstats.Stats('profile_output')
    # stats.sort_stats('cumulative').print_stats(20)

    # text1 = """u don't have to be scared of the loud dog, I'll protect you". The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend.
    # <|endoftext|>"""
    # text2 = """Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
    # Tom asked his friend, Sam, to help him search for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell into the pit."
    # Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."
    # They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear them. They were sad and scared, and they never got out of the pit.
    # <|endoftext|>"""
    # text3 = """Tom and Lily were playing with their toys in the living room. They liked to build towers and bridges with their blocks and cars. Tom was very proud of his tall tower. He wanted to make it even taller, so he reached for more blocks.
    # "Tom, can I have some blocks too?" Lily asked. She wanted to make a bridge for her cars.
    # "No, these are mine. Go find your own," Tom said. He did not want to share with his sister. He pulled the blocks closer to him.
    # Lily felt sad and angry. She did not think Tom was being nice. She looked at his tower and had an idea. She decided to pull one of the blocks at the bottom of the tower.
    # Suddenly, the tower fell down with a loud crash. All the blocks and cars scattered on the floor. Tom and Lily were shocked. They felt the floor shake and heard a rumble. It was an earthquake!
    # "Mommy! Daddy!" they cried. They were scared and ran to their parents, who were in the kitchen.
    # "Are you okay, kids?" Mommy asked. She hugged them and checked if they were hurt.
    # "We're okay, Mommy. But our toys are broken," Lily said.
    # "I'm sorry, Lily. But toys are not important. You are important. We are safe and together. That's what matters," Mommy said.
    # Tom felt sorry for what he did. He realized he was selfish and mean to his sister. He saw how scared she was during the earthquake. He wanted to make her happy.
    # "Lily, I'm sorry I did not share with you. You can have all the blocks you want. I love you, sister," Tom said.
    # Lily smiled and hugged him. She forgave him and thanked him. She loved him too.
    # They went back to the living room and cleaned up their toys. They decided to build something together. They made a big house with a garden and a fence. They put their cars and dolls inside. They were happy and proud of their work.
    # Mommy and Daddy came to see their house. They praised them and gave them a treat. It was a lemon cake. It was sour, but they liked it. They learned that sharing is caring, and that family is sweet.
    # <|endoftext|>"""
    # text4 = """Once upon a time there was a little girl named Lucy. She loved to go to the store to buy sweets with her mom and dad. On this special day, Lucy entered the store with her mom and dad, feeling so excited.
    # As they were looking around, Lucy noticed a little girl playing with a toy in the corner of the store. She gasped in excitement and ran towards her. Lucy asked if she could play too but the little girl said no. She was rather grumpy and was not in the mood to play.
    # Lucy's mom saw what was going on and told Lucy, "Let's try to be peaceful and kind to her. Have patience and understanding. Together, you can both be happy!"
    # So, Lucy smiled at the girl and said, "Can we play together?" The little girl softened and smiled back. She agreed to share the toy and even let Lucy have a turn first.
    # Lucy and the little girl played together happily. In the end, they both learnt an important lesson: be peaceful, kind, and understanding when faced with a conflict. And that is why Lucy and the little girl became great friends.
    # <|endoftext|>"""
    # text5 = """One morning, a cat named Tom woke up. He felt happy because the sun was shining. Tom wanted to start his day, so he did a big stretch. He stretched his legs, his back, and his tail. It felt easy and good.
    # Tom went outside to play. He saw his friend, a dog named Max. Max was also stretching in the morning sun. They both felt very happy. They decided to play together and have fun all day.
    # At the end of the day, Tom and Max were tired. They had played all day and had lots of fun. They said goodbye to each other and went to their homes. Before going to sleep, they both did another easy stretch. Tom knew that tomorrow would be another happy morning.
    # <|endoftext|>"""
    # text6 = """Lily and Tom were twins who liked to decorate things. They had a big box of crayons, stickers, and glitter. One day, they found a shiny copper pot in the kitchen. It was Mom's pot, but she was not home. Lily and Tom wanted to make it more pretty.
    # They took the pot to their room and put it on the floor. They opened their box of crayons, stickers, and glitter. They started to draw and stick and sprinkle on the pot. They made colorful shapes and patterns. They thought the pot looked very nice.
    # But they were clumsy. They did not see that they also made a big mess. They spilled glitter on the floor and the bed. They stuck stickers on the wall and the door. They drew crayons on the window and the dresser. They did not hear Mom come home.
    # Mom saw the mess in the kitchen. She saw the glitter, the stickers, and the crayons. She was angry. She followed the trail to their room. She saw the pot. She saw the floor, the bed, the wall, the door, the window, and the dresser. She was very angry.
    # She said, "Lily and Tom, what did you do? You ruined my pot and my room. You are very naughty. You have to clean up everything. And you have to say sorry."
    # Lily and Tom were scared. They did not mean to make Mom angry. They only wanted to decorate the pot. They said, "Sorry, Mom. We love you. We will clean up. Please don't be mad."
    # Mom sighed. She was still angry, but she also loved them. She said, "I love you too, but you have to be careful. You can't touch my things without asking. And you can't make a mess like this. You have to learn to be more tidy and respectful."
    # Lily and Tom nodded. They hugged Mom and said, "We will, Mom. We will." They took a broom, a dustpan, and a cloth. They started to clean up their mess. They hoped Mom would forgive them. They learned their lesson. They would not decorate Mom's pot again.
    # <|endoftext|>"""
    # text7 = """Once upon a time, there was a king. He was a big and strong king who ruled over his kingdom. One day, he wanted to take a nice and long bath, so he filled up his big bathtub with warm water. He wanted to feel relaxed and so he soaked in the tub for a really long time.
    # When he had finished soaking and stepped out of the bathtub, the king noticed that the water had spilled out of the tub and all over the floor. He felt guilty that he had made such a mess, so he quickly grabbed a cloth and began to clean it up.
    # The king got so hot from cleaning up the mess that he decided to take another soak in the bathtub. He put a lot of bubbles in the water to make it nice and bubbly. He relaxed again and felt all the worries wash away.
    # The king was so happy that he had been able to clean up the mess he had made and enjoy a nice soak. He dried off and wrapped himself up in a big towel. Then, the king went back to ruling his kingdom and enjoying his lovely baths.
    # <|endoftext|>"""
    # text8 = """Lily and Max were playing in the park with their mom. They liked to slide, swing, and run on the grass. They also liked to listen to the birds that made whistles in the trees.
    # "Look, mom, a red bird!" Lily said, pointing to a cardinal.
    # "That's a pretty bird, Lily. Do you know what it is called?" mom asked.
    # "A cardinal, mom. I learned it in school," Lily said proudly.
    # "Very good, Lily. And do you know what that yellow bird is, Max?" mom asked, pointing to a canary.
    # "A canary, mom. I learned it in school, too," Max said.
    # "Wow, you are both very smart. Do you want to learn another bird name?" mom asked.
    # "Yes, mom, yes!" Lily and Max said.
    # "OK, see that blue bird over there? That's a blue jay. It has a very loud whistle. Can you try to whistle like it?" mom asked.
    # Lily and Max tried to whistle, but they only made funny noises. They laughed and mom laughed, too.
    # "Whistling is hard, mom. How do you do it?" Lily asked.
    # "It takes practice, Lily. Maybe when you are older, you can whistle better. But you know what? You don't need to whistle to have fun. You can sing, or clap, or dance, or make any sound you like," mom said.
    # "I like to sing, mom. Can we sing a song?" Max asked.
    # "Sure, Max. What song do you want to sing?" mom asked.
    # "How about 'Twinkle, Twinkle, Little Star'?" Max suggested.
    # "OK, let's sing it together," mom said.
    # They sang the song and looked at the sky. The sun was shining and the clouds were light and fluffy.
    # "That was a nice song, mom. But I'm feeling sleepy now. Can we nap?" Lily asked.
    # "Me too, mom. Can we nap?" Max asked.
    # "Of course, my sweeties. Let's go to the car and nap. You had a busy day," mom said.
    # They walked to the car and mom buckled them in their seats. She gave them each a kiss and a hug.
    # "Sleep well, my loves. I'll wake you up when we get home," mom said.
    # Lily and Max closed their eyes and fell asleep. They dreamed of birds and stars and whistles. They were happy.
    # <|endoftext|>"""
    # text9 = """Once upon a time, there was a big bow. The bow was very strong and reliable. It was the best bow in the town. Everyone liked the bow and wanted to use it. They knew it would help them do their work.
    # One day, a man wanted to test the bow. He was not a good man. He wanted to see if the bow was really strong. He pulled and pulled on the bow. He wanted to see if it would break.
    # The bow did not break because it was strong. But the man did not stop. He pulled harder and harder. At last, the bow broke. The man was not happy. The town was sad. They lost their best bow.
    # <|endoftext|>"""
    # text10 = """Once upon a time, there was a little girl named Lily. She lived in a small, tidy house with her mom, dad, and her dog, Max. Lily loved to play with Max in the backyard. They would run, jump, and have lots of fun together.
    # One day, Lily's mom said, "Lily, I have a special treat for you and Max!" She gave Lily a big, yummy cookie and Max a tasty bone. Lily was very happy and said, "Thank you, mom!" But then, she had an idea. She wanted to save the cookie and the bone for later.
    # Lily put the cookie and the bone in a secret place under her bed. She forgot about them for a few days. When she remembered the treats, she found that the cookie was all broken and the bone was dirty. The treats were spoiled. Lily was sad, but she learned that it's better to enjoy treats when they are fresh and clean.
    # <|endoftext|>"""
    # text_list = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10]
    # tokenizer = Tokenizer.from_files("./checkpoint/BPE", "./checkpoint/BPE", special_tokens=["<|endoftext|>"])
    # total_bytes = 0
    # total_tokens = 0
    # for text in text_list:
    #     bytes_num = len(text.encode("utf-8"))
    #     total_bytes += bytes_num
    #     token_num = len(tokenizer.encode(text))
    #     total_tokens += token_num
    #     print(f"Ratio: {bytes_num / token_num:.4f}")
    # print(f"Average Ratio: {total_bytes / total_tokens:.4f}")
    # total_bytes = 0
    # total_time = 0
    # for text in text_list:
    #     bytes_num = len(text.encode("utf-8"))
    #     start_time = time.time()
    #     ids = tokenizer.encode(text)
    #     end_time = time.time()
    #     print(f"Throughput: {bytes_num / (end_time - start_time):.4f}")
    #     total_bytes += bytes_num
    #     total_time += end_time - start_time
    # print(f"Average Throughput: {total_bytes / total_time:.4f}")

    # tokenize validation
    import numpy as np
    tokenizer = Tokenizer.from_files("./checkpoint/BPE", "./checkpoint/BPE", special_tokens=["<|endoftext|>"])
    with open("../data/TinyStoriesV2-GPT4-valid.txt", "r", encoding="utf-8") as f:
        ids = list(tokenizer.encode_iterable(f))
    arr_valid = np.array(ids, dtype=np.uint16)
    np.save("./checkpoint/BPE/ids_valid.npy", arr_valid)

    with open("../data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
        ids = list(tokenizer.encode_iterable(f))
    arr_train = np.array(ids, dtype=np.uint16)
    np.save("./checkpoint/BPE/ids_train.npy", arr_train)

