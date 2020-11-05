import numpy as np

ZEROS_ = np.zeros(1)
ONES_ = np.ones(1)

FIRST_10_INTS_ = np.arange(10)
FIRST_20_INTS_ = np.arange(20)
FIRST_50_INTS_ = np.arange(50)
FIRST_100_INTS_ = np.arange(100)

MAIN_CONSTANTS = np.array([np.pi, np.e])

FIRST_10_SQUARES_ = np.sqrt(np.arange(10))
FIRST_20_SQUARES_ = np.sqrt(np.arange(20))
FIRST_50_SQUARES_ = np.sqrt(np.arange(50))
FIRST_100_SQUARES_ = np.sqrt(np.arange(100))

NUMBERS_VOCAB = np.unique(
    np.concatenate([FIRST_100_INTS_, FIRST_100_SQUARES_, MAIN_CONSTANTS])
)
