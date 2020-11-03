import unittest
import math

from greippi import vectorizers


class DummyEmbeddingModel:
    def __init__(self, vectors):
        self.vectors = vectors

    def __contains__(self, item):
        return item in self.vectors

    def word_vec(self, token):
        return self.vectors[token]


class TestVectorizers(unittest.TestCase):
    def test_bow_vectorizer(self):
        sentences = [['a', 'b'], ['a']]
        vectorizer = vectorizers.BOWVectorizer(use_tfidf=False)
        vectorizer.fit(sentences)
        vectorized = vectorizer.transform(sentences)
        self.assertEqual(vectorized[0].tolist(), [1, 1])

    def test_tfidf(self):
        sentences = [['a', 'b'], ['a']]
        vectorizer = vectorizers.BOWVectorizer(use_tfidf=True)
        vectorizer.fit(sentences)
        vectorized = vectorizer.transform(sentences)
        self.assertNotEqual(vectorized[0].tolist(), [1, 1])
        self.assertNotEqual(vectorized[0].tolist(), [0, 0])

    def test_mincount(self):
        sentences = [['a', 'b'], ['a']]
        vectorizer = vectorizers.BOWVectorizer(use_tfidf=False, min_count=2)
        vectorizer.fit(sentences)
        vectorized = vectorizer.transform(sentences)
        self.assertEqual(vectorized[0].tolist(), [1])

    def test_mincount(self):
        sentences = [['a', 'b', 'c', 'd', 'e'], ['a']]
        vectorizer = vectorizers.BOWVectorizer(use_tfidf=False, word_count=1)
        vectorizer.fit(sentences)
        vectorized = vectorizer.transform(sentences)
        self.assertEqual(vectorized[0].tolist(), [1])

    def test_compressed_model(self):
        model = DummyEmbeddingModel({'a': [0, 1, 0],
                                     'b': [0, 0, 1]})
        compressed = vectorizers.CompressedEmbeddingModel(model,3)
        compressed.fit(['a b', 'a'])
        embedding_model = vectorizers.EmbeddingModel('avg', None, compressed)
        sentences = embedding_model.transform(['a b'])

        self.assertEqual(sentences.tolist()[0], [0, 1 / math.sqrt(2), 1 / math.sqrt(2)])

    def test_max(self):
        model = DummyEmbeddingModel({'a': [0, 1, 0],
                                     'b': [0, 0, 1]})
        compressed = vectorizers.CompressedEmbeddingModel(model,3)
        compressed.fit(['a b', 'a'])
        embedding_model = vectorizers.EmbeddingModel('max', None, compressed)
        sentences = embedding_model.transform(['a b'])

        self.assertEqual(sentences.tolist()[0], [0, 1, 1])

    def test_sif(self):
        model = DummyEmbeddingModel({'a': [0, 0, 0],
                                     'b': [0, 0, 1]})
        compressed = vectorizers.CompressedEmbeddingModel(model, 3)
        compressed.fit(['a b', 'a'])
        embedding_model = vectorizers.EmbeddingModel('sif', None, compressed)
        embedding_model.fit(['a b', 'a'])
        sentences = embedding_model.transform(['a b'])

        self.assertEqual(sentences.tolist()[0], [0, 0, 0])
