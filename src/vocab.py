import pickle

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.idx = 4

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def numericalize(self, sentence):
        return [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in sentence.split()]

    def decode(self, indices):
        return " ".join([self.idx2word.get(i, "<UNK>") for i in indices])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.word2idx)
