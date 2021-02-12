import numpy as np


INF = 10 ** 4


START_OF_SEQUENCE = 'sos'
END_OF_SEQUENCE = 'eos'
PADDING = 'pad'
NUMBER_SYMBOL = '<n>'
VARIABLES = {'x'}
NUMBERS = {
    # '1', '2', '3',
}


class Operator:
    def __init__(self, arity, name, f_eval, f_repr=None):
        self.arity = arity
        self.name = name
        self.eval = f_eval
        if f_repr is None:
            f_repr = lambda params: f"{self.name}({','.join(p for p in params)})"
        self.repr = f_repr


OPERATORS = {
    'cos': Operator(1, 'np.cos',
                    lambda params: np.cos(params[0])),
    'sin': Operator(1, 'np.sin',
                    lambda params: np.sin(params[0])),
    'add': Operator(2, 'add',
                    lambda params: params[0] + params[1],
                    lambda params: f"({params[0]}) + ({params[1]})"),
    'mult': Operator(2, 'mult',
                     lambda params: params[0] * params[1],
                     lambda params: f"({params[0]}) * ({params[1]})"),
    # 'pow': Operator(2, 'pow',
    #                  lambda params: params[0] ** params[1],
    #                  lambda params: f"({params[0]}) ** ({params[1]})"),
}

INDEX_TO_TOKEN = [
    START_OF_SEQUENCE,
    END_OF_SEQUENCE,
    PADDING,
    NUMBER_SYMBOL,
    *NUMBERS,
    *VARIABLES,
    *OPERATORS.keys(),
]
TOKEN_TO_INDEX = {
    t: i for i, t in enumerate(INDEX_TO_TOKEN)
}


if __name__ == '__main__':
    print(TOKEN_TO_INDEX)
    print(INDEX_TO_TOKEN)
