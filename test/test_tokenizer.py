import unittest

from greippi import tokenizer

class TokenizerTest(unittest.TestCase):
    def test_tokenization(self):
        self.assertEqual(tokenizer.produce_tokens('Tämä on testi, jossa on kaksi lausetta.'),
                         ['Tämä', 'on', 'testi', 'jossa', 'on', 'kaksi', 'lausetta'])

    def test_tokenization_lowercase(self):
        self.assertEqual(tokenizer.produce_tokens('Tämä on testi.', lowercase=True), ['tämä', 'on', 'testi'])

    def test_tokenization_lemmatize(self):
        self.assertEqual(tokenizer.produce_tokens('Tämä on testi.', lowercase=True, normalization='lemma'),
                         ['tämä', 'olla', 'testi'])

    def test_tokenization_stem(self):
        self.assertEqual(tokenizer.produce_tokens('Tämä on testi, joka sisältää sanoja joissa on suffikseja.',
                                                  lowercase=True, normalization='stem'),
                         ['tämä', 'on', 'test', 'joka', 'sisält', 'sano', 'jois', 'on', 'suffiks'])

    def test_tokenization_stopword(self):
        self.assertEqual(tokenizer.produce_tokens('Tämä on testi, joka poistaa poistosanat lopputuloksesta.',
                                                  lowercase=True, remove_stopwords=True),
                         ['testi', 'poistaa', 'poistosanat', 'lopputuloksesta'])

    def test_produce_text(self):
        variables = [{
            'a': '1234',
            'b': '2345',
            'c': '3456'
        }, {
            'a': 'a',
            'b': '2345',
            'c': 'c'
        }]

        fields = ['a', 'c']

        self.assertEqual(tokenizer.produce_text(variables, fields), ['1234 3456', 'a c'])