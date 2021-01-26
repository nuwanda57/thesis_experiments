import formulas_vae_v3.formula_config as my_formula_config


class Vocab(object):
    def __init__(self):
        self.token_to_index = {}
        self.index_to_token = []

        self.unknown_token = '<unk>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'

        self.service_tokens = {self.unknown_token, self.sos_token, self.eos_token, self.pad_token}

        self.index_to_token.append(self.unknown_token)
        self.index_to_token.append(self.sos_token)
        self.index_to_token.append(self.eos_token)
        self.index_to_token.append(self.pad_token)

        self.token_to_index[self.unknown_token] = 0
        self.token_to_index[self.sos_token] = 1
        self.token_to_index[self.eos_token] = 2
        self.token_to_index[self.pad_token] = 3

        self.unknown_token_index = self.token_to_index['<unk>']
        self.sos_token_index = self.token_to_index['<sos>']
        self.eos_token_index = self.token_to_index['<eos>']
        self.pad_token_index = self.token_to_index['<pad>']

        self.constant_tokens = set()
        self.unary_operator_tokens = set()
        self.binary_operator_tokens = set()
        self.variable_tokens = set()
        self.constant_indices = set()
        self.unary_operator_indices = set()
        self.binary_operator_indices = set()
        self.variable_indices = set()

        self.number_start_symbol = None

        self._is_built = False

    def size(self):
        assert self._is_built
        return len(self.index_to_token)

    def build_vocab_from_formula_config(self):
        assert not self._is_built, 'vocab is already built'
        for token in my_formula_config.VARIABLES:
            self.variable_tokens.add(token)
            self.token_to_index[token] = len(self.token_to_index)
            self.variable_indices.add(len(self.token_to_index))
            self.index_to_token.append(token)
        for token in my_formula_config.NUMBERS:
            self.constant_tokens.add(token)
            self.token_to_index[token] = len(self.token_to_index)
            self.constant_indices.add(len(self.token_to_index))
            self.index_to_token.append(token)
        for token in my_formula_config.OPERATORS_1:
            self.unary_operator_tokens.add(token)
            self.token_to_index[token] = len(self.token_to_index)
            self.unary_operator_indices.add(len(self.token_to_index))
            self.index_to_token.append(token)
        for token in my_formula_config.OPERATORS_2:
            self.binary_operator_tokens.add(token)
            self.token_to_index[token] = len(self.token_to_index)
            self.binary_operator_indices.add(len(self.token_to_index))
            self.index_to_token.append(token)
        token = my_formula_config.NUMBER_START_SYMBOL
        self.number_start_symbol = token
        self.token_to_index[token] = len(self.token_to_index)
        self.index_to_token.append(token)
        self._is_built = True

    def write_vocab_to_file(self, vocab_file):
        assert self._is_built, 'vocab is not built'
        with open(vocab_file, 'w') as f:
            f.write('\n'.join(self.index_to_token))
