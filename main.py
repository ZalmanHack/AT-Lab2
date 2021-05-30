import inspect
import json
import os
import time

from dict_adapter.sentences import Sentences
from dict_adapter.sentence_builder import SentencesBuilder


def run(dictAdapterSentence: Sentences):
    while True:
        base_sentence = input('Введите слово: ')
        os.system('cls')
        if base_sentence == 'q':
            break
        print('Ваша фраза:  {}\n'.format(base_sentence))
        start_time = time.time()
        find_probability, find_perplexity = dictAdapterSentence.find(base_sentence, testing=False)
        end_time = time.time()
        print(f'\nВремя выполнения: {end_time - start_time:.3f} сек.')
        print(f'Вероятность:      {find_probability}')
        print(f'Перплексия:       {find_perplexity}\n')

        gen_probability, gen_perplexity, gen_sentence = dictAdapterSentence.generate(base_sentence, testing=False)
        print(f'\nВремя выполнения: {end_time - start_time:.3f} сек.')
        print(f'Вероятность:      {gen_probability}')
        print(f'Перплексия:       {gen_perplexity}\n')
        print(f'Генерация текста: {gen_sentence}')


def run_testing(adapter):
    adapter.testing()


def open_settings():
    try:
        with open('settings.json', 'r', encoding='utf-8') as file:
            data = json.loads('\n'.join([line for line in file]))
            _ngram_len = int(data["ngram_len"])
            _alpha = float(data["alpha"])
            _lambdas = list(data['lambdas'])
            assert _ngram_len == len(_lambdas), 'Количество лямбд должно соответствовать размеру N-грамм'
            return _ngram_len, _alpha, _lambdas
    except Exception as _e:
        print(inspect.stack()[0][3], ':', 'Не удалось прочесть файл конфигурации!')
        print(inspect.stack()[0][3], ':', _e)


if __name__ == '__main__':
    try:
        os.system('cls')
        dictAdapter = None
        while True:
            print("1 | загрузить словарь")
            print("2 | создать словарь")
            print("3 | запуск")
            print("4 | тестирование")
            print("5 | выйти")
            cin = input(">> ")
            os.system('cls')

            ngram_len, alpha, lambdas = open_settings()

            if cin == "1":
                path = input("Введите имя файла: ")
                dictAdapter = SentencesBuilder() \
                    .open(path) \
                    .build()
            elif cin == "2":
                path = input("Введите имя файла: ")
                dictAdapter = SentencesBuilder() \
                    .set_ngram_len(ngram_len) \
                    .set_alpha(alpha) \
                    .set_lambdas(lambdas) \
                    .load_texts('.\\text corpora', encoding='utf-8') \
                    .fit() \
                    .save(path) \
                    .build()
            elif cin == "3":
                if isinstance(dictAdapter, Sentences):
                    run(dictAdapter)
            elif cin == "4":
                if isinstance(dictAdapter, Sentences):
                    run_testing(dictAdapter)
            elif cin == "5":
                break
    except Exception as e:
        print(inspect.stack()[0][3], ':', e)
