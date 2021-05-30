

class Model:
    @staticmethod
    def __get_ngrams_joined(sentences, ngram_len):
        ngrams = []
        for sentence in sentences:
            n_gram_count = len(sentence) - ngram_len + 1
            ngrams.append(' '.join([sentence[pos:pos + ngram_len] for pos in range(n_gram_count)]))
        return ngrams

    @staticmethod
    def __count(words_seq: str, full_text: str):
        count = 0
        start = -1
        while True:
            start = full_text.find(words_seq, start + 1)
            if start < 0:
                return count
            count += 1

    @staticmethod
    def laplace(words_seq, full_text, alpha, unique_words):
        return (Model.__count(' '.join(words_seq),
                              full_text) + alpha) / (Model.__count(' '.join(words_seq[:-1]),
                                                                   full_text) + alpha * len(unique_words))

    @staticmethod
    def backward(words_seq, full_text, alpha, unique_words, lambdas):
        return sum([_lambda * Model.laplace(words_seq[:-index], full_text, alpha,
                                            unique_words) for index, _lambda in enumerate(lambdas)])
