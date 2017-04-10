def get_n_gram(text, n):
    n_grams = []
    for i in range(len(text) - n + 1):
        n_gram = text[i:i + n]
        n_grams.append(n_gram)
    return n_grams
