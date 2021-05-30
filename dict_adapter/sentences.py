import glob
import inspect
import json
import os
import random
import re
import secrets

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from nlp.metrics import Metrics
from nlp.model import Model
from custom_console import custom_console


class Sentences:
    def __init__(self):
        self._ngram_len: int = 2
        self._alpha: float = 0.2
        self._lambdas = []
        self._test_text = []
        self._train_text = []

        self._dict_data: dict = {}
        self._sentences_data = []
        self._unique_words = {}
        self._clear_text = ''

    @property
    def ngram_len(self) -> int:
        return self._ngram_len

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def lambdas(self) -> list:
        return self._lambdas

    @property
    def test_text(self) -> list:
        return self._test_text

    @property
    def train_text(self) -> list:
        return self._train_text

    @property
    def dict_data(self) -> dict:
        return self._dict_data

    @property
    def sentences_data(self) -> list:
        return self._sentences_data

    @property
    def unique_words(self) -> dict:
        return self._unique_words

    @property
    def clear_text(self) -> str:
        return self._clear_text

    @ngram_len.setter
    def ngram_len(self, value: int) -> None:
        if self._ngram_len == value:
            return
        self._ngram_len = value

    @alpha.setter
    def alpha(self, value: float) -> None:
        if self._alpha == value:
            return
        self._alpha = value

    @lambdas.setter
    def lambdas(self, value: list) -> None:
        if self._lambdas == value:
            return
        self._lambdas = value

    @test_text.setter
    def test_text(self, value: list) -> None:
        if self._test_text == value:
            return
        self._test_text = value

    @train_text.setter
    def train_text(self, value: list) -> None:
        if self._train_text == value:
            return
        self._train_text = value

    @dict_data.setter
    def dict_data(self, value: dict) -> None:
        if self._dict_data == value:
            return
        self._dict_data = value

    @sentences_data.setter
    def sentences_data(self, value: list) -> None:
        if self._sentences_data == value:
            return
        self._sentences_data = value

    @unique_words.setter
    def unique_words(self, value: dict):
        if self._unique_words == value:
            return
        self._unique_words = value

    @clear_text.setter
    def clear_text(self, value: dict):
        if self._clear_text == value:
            return
        self._clear_text = value

    @ngram_len.deleter
    def ngram_len(self) -> None:
        del self._ngram_len

    @alpha.deleter
    def alpha(self) -> None:
        del self._alpha

    @lambdas.deleter
    def lambdas(self) -> None:
        del self._lambdas

    @test_text.deleter
    def test_text(self) -> None:
        del self._test_text

    @train_text.deleter
    def train_text(self) -> None:
        del self._train_text

    @dict_data.deleter
    def dict_data(self) -> None:
        del self._dict_data

    @sentences_data.deleter
    def sentences_data(self) -> None:
        del self._sentences_data

    @unique_words.deleter
    def unique_words(self) -> None:
        del self._unique_words

    @clear_text.deleter
    def clear_text(self) -> None:
        del self._clear_text

    def __load_train_text(self):
        # получение уникального списка слов
        for sentence in tqdm(self.train_text, desc=f"Подсчет уникальных слов"):
            words = self.__get_words(sentence)
            # добавляем слова (сохранятся только уникальные) в массив и подсчитываем их количество
            for word in words:
                if word not in self.unique_words:
                    self.unique_words[word] = 0
                self.unique_words[word] += 1
            if len(words) > 0:
                self.sentences_data.append(' '.join(self.__get_words(sentence)))
        self.clear_text = ' '.join(self.sentences_data)

    def __load_test_text(self):
        for index, sentence in enumerate(self.test_text):
            tmp_sentence = self.__get_words(sentence)
            if len(tmp_sentence) == 0:
                continue
            index_word = random.randint(self.ngram_len - 1, len(tmp_sentence) - self.ngram_len)
            tmp_sentence[index_word] = secrets.choice(list(self.unique_words.keys()))
            self.test_text[index] = ' '.join(tmp_sentence)

    def __get_words(self, sentence, start=True, end=True):
        result = []
        if start is True:
            result = ['$'] * (self.ngram_len - 1)
        words = re.findall('[а-яёa-z]+', sentence.lower())
        if not words:
            return []
        result += words
        if end is True:
            result += ['$'] * (self.ngram_len - 1)
        return result

    def __get_ngrams(self, sentence):
        n_gram_count = len(sentence) - self.ngram_len + 1
        return [sentence[pos:pos + self.ngram_len] for pos in range(n_gram_count)]

    def __find_in_vac(self, sentence):
        bit = '|'.join(sentence[:-1])
        if bit in self.dict_data and sentence[-1] in self.dict_data[bit]:
            return self.dict_data[bit][sentence[-1]], True
        else:
            return Model.backward(sentence, self.clear_text, self.alpha, self.unique_words, self.lambdas), False

    def __generate_text(self, sentence: list):
        if len(sentence) == self.ngram_len:
            bit = '|'.join(sentence[1:])
            if bit in self.dict_data:
                answer = list(sorted([[word, value] for word, value in self.dict_data[bit].items()],
                                     key=lambda l: l[1], reverse=True))[0]
                sentence.append(answer[0])
                return sentence[1:], answer[1], True
        return sentence, 0, False

    def __assessment_existing(self, ngrams_list, testing=False):
        probability = 1
        index = 0
        while index < len(ngrams_list):
            # поиск вероятности в словаре
            answer = self.__find_in_vac(ngrams_list[index])
            if not testing:
                print(custom_console.build_row([ngrams_list[index], answer[0], answer[1]], max_width=45))
            probability *= answer[0]
            index += 1
        return probability

    def __generation(self, ngram, testing=False):
        probability = 1
        new_probability = 1
        full_sentence = ''
        while new_probability != 0:
            ngram, new_probability, answer = self.__generate_text(ngram)
            if ngram[1:] == ['$'] * (self.ngram_len - 1):
                break
            if new_probability != 0:
                full_sentence += ' ' + ngram[-1]
                if not testing:
                    print(custom_console.build_row([ngram, new_probability, answer], max_width=45))
                probability *= new_probability
        return probability, full_sentence

    def load_texts(self, dir_path: str, encoding: str) -> None:
        text_data = ''
        for path in glob.glob(os.path.join(dir_path, '*.txt')):
            with open(path, 'r', encoding=encoding) as file:
                text_data += file.read()
        sentences = list(filter(None, re.split(' *[.?!][\'")\]]* *', text_data.replace('\n', ' '))))
        self.train_text, self.test_text = train_test_split(sentences, shuffle=True, test_size=0.30, random_state=42)
        self.__load_train_text()
        self.__load_test_text()

    def fit(self):
        ngrams = []
        for sentence in self.sentences_data:
            ngrams.extend(self.__get_ngrams(self.__get_words(sentence)))
        for words_seq in tqdm(ngrams, desc='Обучение модели'):
            probability: float = Model.backward(words_seq, self.clear_text, self.alpha, self.unique_words, self.lambdas)
            ngram_key = '|'.join(words_seq[:-1])
            if ngram_key in self.dict_data:
                self.dict_data[ngram_key][words_seq[-1]] = probability
            else:
                self.dict_data[ngram_key] = {words_seq[-1]: probability}

    def find(self, sentence, testing=False):
        words_list = self.__get_words(sentence, start=not testing, end=False)
        ngrams_list = self.__get_ngrams(words_list)
        probability = self.__assessment_existing(ngrams_list, testing=testing)
        perplexity = Metrics.perplexity(probability, len(words_list))
        return probability, perplexity

    def generate(self, sentence, testing=False):
        probability, _ = self.find(sentence, testing)

        words_list = self.__get_words(sentence, start=not testing, end=False)
        ngrams_list = self.__get_ngrams(words_list)

        new_probability, new_sentence = self.__generation(ngrams_list[-1], testing=testing)
        result_sentence = sentence + new_sentence
        result_probability = probability * new_probability

        words_list = self.__get_words(result_sentence, start=not testing, end=False)
        perplexity = Metrics.perplexity(result_probability, len(words_list))
        return result_probability, perplexity, result_sentence

    def testing(self):
        probability_list = []
        perplexity_list = []
        print(custom_console.build_row(['№', '|',
                                        custom_console.build_row(['Вероятность', 'Перплексия', 'Предложение'], 30)], 4))
        print('-' * (30 * 3 + 4))
        for index, sentence in enumerate(self.test_text):
            probability, perplexity = self.find(sentence, testing=True)
            print(custom_console.build_row([index + 1, '|',
                                            custom_console.build_row([probability, perplexity, sentence], 30)], 4))
            probability_list.append(probability)
            perplexity_list.append(perplexity)
        print(f'Усредненная вероятность: {Metrics.get_std(probability_list)}')
        print(f'Усредненная перплексия:  {Metrics.get_std(perplexity_list)}')

    def save(self, file_name):
        try:
            with open(os.path.join('.\\dictionaries', 'train_{0}.json'.format(file_name)), 'w', encoding='utf-8') as f:
                f.write(json.dumps({
                    "alpha": self.alpha,
                    "ngram_len": self.ngram_len,
                    "lambdas": self.lambdas,
                    "unique_words": self.unique_words,
                    "dict_data": self.dict_data,
                    "sentences": self.sentences_data},
                    indent=4, ensure_ascii=False))
            with open(os.path.join('.\\dictionaries', 'test_{0}.json'.format(file_name)), 'w', encoding='utf-8') as f:
                f.write(json.dumps({"sentences": self.test_text}, indent=4, ensure_ascii=False))
        except Exception as e:
            print(inspect.stack()[0][3], ':', 'Не удалось сохранить словари!')
            print(inspect.stack()[0][3], ':', e)

    def open(self, file_name) -> None:
        try:
            with open(os.path.join('.\\dictionaries', 'train_{0}.json'.format(file_name)), 'r',
                      encoding='utf-8') as f:
                data = json.loads('\n'.join([line for line in tqdm(f, desc='Чтение словаря "train"')]))
                self.alpha = float(data["alpha"])
                self.ngram_len = int(data["ngram_len"])
                self.lambdas = list(data["lambdas"])
                assert self.ngram_len == len(self.lambdas), 'Количество лямбд должно соответсвовать размеру N-грам'
                self.unique_words = dict(data["unique_words"])
                self.dict_data = data["dict_data"]
                self.sentences_data = data["sentences"]
                self.clear_text = ' '.join(self.sentences_data)
            with open(os.path.join('.\\dictionaries', 'test_{0}.json'.format(file_name)), 'r',
                      encoding='utf-8') as f:
                data = json.loads('\n'.join([line for line in tqdm(f, desc='Чтение словаря "test"')]))
                self.test_text = list(data['sentences'])
        except Exception as e:
            print(inspect.stack()[0][3], ':', 'Не удалось прочесть словари!')
            print(inspect.stack()[0][3], ':', e)
