from enum import Enum

VAR_TYPE = Enum('Variable Type', ['RAND_CHAR', 'RAND_STR','TUBINGEN'])
C_LADDERS = [ 'obs', 'inter', 'sub']
DATASETS = [ 'rchar', 'cs', 'adv' ]
TASKS = 3


# template prompts related
LABEL = 'label'
PRED = 'pred'
PROMPT = 'prompt'
QUESTION = 'question'
RESPONSE = 'response'
R_PRED = 'r_pred'

# prompt format
NO_COT_INPUT_PROMPT = f"%s. \n%s"

# intervention
OBS_INTER_MAP = [
    [0, 1, 0, 1], # task 0
    [0, 1, 3, 0, 1, 3, 0, 1, 3], # task 1
    [0, 1, 3, 0, 1, 3, 0, 1, 3], # task 2
]

# OBS_INTER_MAP = [
#     [0, 0], # task 0
#     [0, 1, 3, 0, 2, 1, 1, 4, 0], # task 1
#     [0, 3, 4, 0, 3, 1, 1, 0, 3], # task 2
# ]

NUM_INTERS = [
    2, # task 0
    3, # task 1
    3, # task 2
]
INTER_VARS = [
    ['A', 'B'],
    ['A', 'B', 'C'],
    ['A', 'B', 'C']
]

# Relevant Query indicies:

# 1. NS (Random Char)
C0_NS = { # no context influence
    0: ([1, 3], []),
    1: ([2, 5, 8], []),
    2: ([1, 4, 7], [])
}

C1_NS = { # only context influence
    0: ([0], [2]),
    1: ([0, 1, 4, 6], [3, 7]),
    2: ([0, 2, 5, 6], [3, 8])
}

M1_NS = { # only memory influence
    0: ([], []),
    1: ([], []),
    2: ([], [])
}

# 2. CS (Tubingen)
C0_CS = C0_NS # no context influence
C1_CS = C1_NS # only context influence
M1_CS = C1_NS # only memory influence

# 3. ACS (Adversarial Tubingen)
CO_MO_ACS = { # no context or memory influence
    0: ([], []),
    1: ([], []),
    2: ([], [])
}

C0_M0_ACS = { # no context or memory influence.
    0: ([], []),
    1: ([], []),
    2: ([], [])
}


C1_MO_ACS = { # only context influence
    0: ([0], [2]),
    1: ([0, 1, 4, 6], [3, 7]),
    2: ([0, 2, 5, 6], [3, 8]),
}

C0_M1_ACS = { # only memory influence
   0: ([], [1, 3]),
    1: ([], [2, 5, 7]),
    2: ([1], [4, 7])
}

C1_M1_ACS = { # both context and memory influence.
    0: ([], []),
    1: ([], []),
    2: ([], [])
}
