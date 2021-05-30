import statistics
from math import sqrt


class Metrics:
    @staticmethod
    def get_std(x: list) -> float:
        x_mean = statistics.mean(x)
        return sqrt(sum([pow(item - x_mean, 2) for item in x]) / (len(x) - 1))

    @staticmethod
    def perplexity(probability, sentence_len) -> float:
        try:
            return pow(probability, -1 / sentence_len)
        finally:
            return 0.0
