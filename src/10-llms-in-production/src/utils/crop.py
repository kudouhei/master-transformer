"""Byte pair encoding utilities (GPT-2中常用的子词分词算法)"""

import json
import os
import regex as re
import requests

from functools import lru_cache
from tqdm import tqdm

def _get_encoder(subdir):
    """
    Load the encoder from the subdirectory.
    """
    for filename in ["encoder.json", "vocab.bpe"]:
        r = requests.get(
            "https://openaipublic.blob.core.windows.net/gpt-2/"
            + subdir
            + "/"
            + filename,
            stream=True,
        )
        with open(os.path.join(subdir, filename), "wb") as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc=f"Downloading {filename}", total=file_size, unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


###
# 函数功能
# 输入：一个由符号组成的元组 word，例如 ('h', 'e', 'l', 'l', 'o')（表示单词 "hello"）。
# 输出：所有相邻符号对的集合，例如 {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o')}
###
def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    def __init__(self, encoder, bpe_merges, errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))

            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        
        word = "".join(word)

        self.cache[token] = word
        return word
    
    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )
            bpe_tokens.extend(
                self.encoder[bpe_token]
                for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens
    
            
    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=self.errors
        )
        return text
    

def get_encoder(model_name):
    subdir = os.path.join("models", model_name)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    if not os.path.exists(os.path.join(subdir, "encoder.json")):
        _get_encoder(subdir)

    subdir = subdir.replace("\\", "/")  # needed for Windows

    with open(os.path.join(subdir, "encoder.json"), "r") as f:
        encoder = json.load(f)
    with open(
        os.path.join(subdir, "vocab.bpe"), "r", encoding="utf-8"
    ) as f:
        bpe_data = f.read()
    bpe_merges = [
        tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]
    ]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )


enc = get_encoder("124M")

def crop_prompt(prompt: str):
    global enc

    cropped_prompt = enc.decode(enc.encode(prompt)[:2048])
    return cropped_prompt


def crop(s):
    prompt = crop_prompt(s)
    return prompt