from collections import Counter


class Vocab(object):
    def __init__(self):
        self.token_to_index = {}
        self.index_to_token = []

        self.unknown_token = '<unk>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'

        self.service_tokens = {self.unknown_token, self.sos_token, self.eos_token}

        self.index_to_token.append(self.unknown_token)
        self.index_to_token.append(self.sos_token)
        self.index_to_token.append(self.eos_token)

        self.token_to_index[self.unknown_token] = 0
        self.token_to_index[self.sos_token] = 1
        self.token_to_index[self.eos_token] = 2

        self._is_built = False

    def size(self):
        return len(self.index_to_token)

    def build_from_vocab_file(self, vocab_path):
        assert not self._is_built, 'vocab is already built'
        with open(vocab_path) as f:
            for line in f:
                tok = line.split()[0]
                assert tok not in self.service_tokens, 'vocab file contains service tokens'
                self.token_to_index[tok] = len(self.token_to_index)
                self.index_to_token.append(tok)
        self._is_built = True

    def _build_from_counter(self, tokens_counter):
        assert not self._is_built, 'vocab is already built'
        for tok, _ in tokens_counter.most_common():
            assert tok not in self.service_tokens, 'vocab file contains service tokens'
            self.token_to_index[tok] = len(self.token_to_index)
            self.index_to_token.append(tok)
        self._is_built = True

    def build_from_formula_list(self, formulas):
        # formulas: [['2', 'x', '*'], ['x', '4', '^'], ['3', '4']]
        tokens = [tok for f in formulas for tok in f]
        tokens_counter = Counter(tokens)
        self._build_from_counter(tokens_counter)

    def build_from_formula_file(self, formulas_path):
        tokens_counter = Counter()
        with open(formulas_path) as f:
            for formula in f:
                tokens_counter.update(formula.split())
        self._build_from_counter(tokens_counter)

    def write_vocab_to_file(self, vocab_file):
        assert self._is_built, 'vocab is not built'
        with open(vocab_file, 'w') as f:
            f.write('\n'.join(self.index_to_token[len(self.service_tokens):]))
