
import regex as re 
from collections import defaultdict

from concurrent.futures import ProcessPoolExecutor
import regex as re

def train_bpe(input_path, vocab_size, special_tokens):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    escaped = [re.escape(tok) for tok in special_tokens]
    special_pat = "(" + "|".join(escaped) + ")"
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read() 
    docs= re.split(special_pat, content)

    merges = []

    vocab = []
    vocab += special_tokens
    #词汇表应该如何初始化？
    for i in range(256):
        b = bytes([i])
        vocab.append(b)

    corpus = []
    for text in docs:
        if text in special_tokens:
            corpus.append(text)
        else:
            for m in re.finditer(PAT, text):
                corpus.append(m.group())

    word_freq = defaultdict(int)

    for word in corpus:
        byte_tuple = tuple(bytes([b]) for b in word.encode('utf-8'))

        word_freq[byte_tuple] += 1

    word_keys = list(word_freq.keys())


    idx_freq = defaultdict(int)
    for i in range(len(word_keys)):
        key = word_keys[i]
        idx_freq[i] = word_freq[key]


    pair_freq = defaultdict(int)
    source_info = defaultdict(set)

    for idx, value in idx_freq.items():
        key = word_keys[idx]

        for i in range(len(key)-1):
            pair = (key[i],key[i+1])
            pair_freq[pair] += value
            source_info[pair].add(idx)
    while len(vocab) < vocab_size:

        if not pair_freq: break
        

        best_pair = max(pair_freq, key=lambda pair: (pair_freq[pair], pair))
        merges.append(best_pair)
        vocab.append(best_pair[0]+best_pair[1])

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

                if i<len(word)-1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(best_pair[0]+best_pair[1])
                    position.append(i)          
                    new_position.append(i-off_set)
                    off_set += 1

                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)

            word_keys[idx] = new_word
            old_pair = {best_pair}
            for i in position:
                if i>0:
                    old_pair.add((word[i-1],word[i]))
                
                if i+1<len(word)-1:
                    old_pair.add((word[i+1],word[i+2]))
            new_word_pairs = set()
            for i in range(len(new_word)-1):
                new_word_pairs.add((new_word[i], new_word[i+1]))
            
        
            for pair in old_pair:
                if pair not in new_word_pairs:
                    pair_freq[pair] -= value
                    source_info[pair].discard(idx)
                    if pair_freq[pair]<0:
                        del pair_freq[pair]
            add_pair = set()
            for i in new_position:
                if i>0:
                    add_pair.add((new_word[i-1],new_word[i]))
                    
                if i<len(new_word)-1:
                    add_pair.add((new_word[i],new_word[i+1]))
            for pair in add_pair:
                pair_freq[pair] += value
                source_info[pair].add(idx)
    vocab_dict = {i: vocab[i] for i in range(len(vocab))}
    return vocab_dict, merges
import time
if __name__=="__main__":
    input_path = '../data/TinyStories-valid.txt'
    special_tokens = ["<|endoftext|>"]
    s = time.time()
    vocab, merges = train_bpe(input_path, 1000, special_tokens)
    e = time.time()
    print('use time', e - s)
    print(vocab[:30],merges[:30])

