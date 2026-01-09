
from collections import defaultdict
text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
corpus = text.split('')

vocab = ["<|endoftext|>"]
merges = []
for i in range(256):
    b = bytes([i])
    vocab.append(b)

word_freq = defaultdict(int)
for word in corpus:
    byte_tuple = tuple(bytes([b]) for b in word.encode('utf-8'))
    word_freq[byte_tuple] += 1

while len(vacab) < 10:
    pair_freq = defaultdict(int)
    for key, value in word_freq.items():

        for i in range(len(key)-1):
            pair = (key[i],key[i+1])
            pair_freq[pair] += value
    if not pair_freq:
        break

    best_pair = max(pair_freq, key=lambda pair: (pair_freq[pair], pair))
    merges.append(best_pair)

    vocab.append(best_pair[0]+best_pair[1])


    new_byte_freq = {}
    for key, value in word_freq.items():
        new_key = []
        i = 0
        while i < len(key):

            if i<len(key)-1 and key[i] == best_pair[0] and key[i+1] == best_pair[1]:
                new_key.append(best_pair[0]+best_pair[1])
                i += 2
            else:
                new_key.append(key[i])
                i += 1
        new_key = tuple(new_key)
        new_byte_freq[new_key] = new_byte_freq.get(new_key, 0) + value
    word_freq = new_byte_freq
print(vocab)
print(merges)





