from utils.terms import REFUTING_TERMS

def get_refuting_feature(headline):
    result = [1 if term in headline else 0 for term in REFUTING_TERMS]
    return result


