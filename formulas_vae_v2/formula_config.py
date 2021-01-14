import numpy as np


NUMBER_START_SYMBOL = '<n>'
VARIABLES = {'x'}
COEFFICIENTS = [str(i) for i in np.arange(50)]
OPERATORS = {
    'add': 2,
    'mult': 2,
    'pow': 2
}
