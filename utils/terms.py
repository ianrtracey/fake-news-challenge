hard_coded_refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
]

more_refuting_words = [
    'contrary',
    'unlike',
    'contradictory',
    'contradiction',
    'different',
    'divergent',
    'unsimilar',
    'antithetical',
    'opposite',
    'mismatched',
    'mismatch',
]


hard_coded_hedge_words = [
        "argue",
        "argument",
        "believe",
        "belief",
        "conjecture",
        "consider",
        "hint",
        "hypothesis",
        "hypotheses",
        "hypothesize",
        "implication",
        "imply",
        "indicate",
        "predict",
        "prediction",
        "previous",
        "previously",
        "proposal",
        "propose",
        "question",
        "speculate",
        "speculation",
        "suggest",
        "suspect",
        "theorize",
        "theory",
        "think",
        ]


with open("utils/hedge_words.txt") as file:
    hedge_words = file.readlines()
    cleaned_hedge_words = [word.rstrip() for word in hedge_words]
    HEDGING_WORDS = hard_coded_hedge_words + cleaned_hedge_words

REFUTING_TERMS = hard_coded_hedge_words + more_refuting_words

