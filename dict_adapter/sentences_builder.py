from __future__ import annotations

from dict_adapter.sentences import Sentences


class SentencesBuilder:
    def __init__(self):
        self.dictAdapter = Sentences()

    def load_texts(self, dir_path: str, encoding: str) -> SentencesBuilder:
        self.dictAdapter.load_texts(dir_path, encoding)
        return self

    def set_ngram_len(self, value: int) -> SentencesBuilder:
        self.dictAdapter.ngram_len = value
        return self

    def set_alpha(self, value: float) -> SentencesBuilder:
        self.dictAdapter.alpha = value
        return self

    def set_lambdas(self, value: list) -> SentencesBuilder:
        self.dictAdapter.lambdas = value
        return self

    def fit(self) -> SentencesBuilder:
        self.dictAdapter.fit()
        return self

    def save(self, file_name: str) -> SentencesBuilder:
        self.dictAdapter.save(file_name)
        return self

    def open(self, file_name) -> SentencesBuilder:
        self.dictAdapter.open(file_name)
        return self

    def build(self) -> Sentences:
        return self.dictAdapter
