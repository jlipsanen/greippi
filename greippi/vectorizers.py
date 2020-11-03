import os

import fasttext
import fasttext.util
import gensim.models.word2vec
import gensim.models.fasttext
import numpy as np

from greippi.utils import produce_token_counts
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from greippi import tokenizer, utils


def get_token_embedding(token, vectors):
    if isinstance(vectors, fasttext.FastText._FastText):
        return vectors.get_word_vector(token)
    can_use_oov_words = isinstance(vectors, gensim.models.keyedvectors.FastTextKeyedVectors)
    if token in vectors or can_use_oov_words:
        return np.copy(vectors.word_vec(token))
    return None


def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X, pc):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    XX = X - X.dot(pc.transpose()) * pc

    return XX


class BOWVectorizer:
    def __init__(self, word_count=None, min_count=None, use_tfidf=True):
        self.tfidf_transformer = None
        self.words = {}
        self.word_count = word_count
        self.min_count = min_count
        self.use_tfidf = use_tfidf

    def fit(self, sentences):
        token_counts = produce_token_counts(sentences)
        inverse_order = sorted(token_counts.items(), key=lambda x: -x[1])

        for index, token_pair in enumerate(inverse_order):
            token = token_pair[0]
            count = token_pair[1]
            if (self.min_count is not None and count < self.min_count)\
                    or (self.word_count is not None and index >= self.word_count):
                break
            self.words[token] = index

        if not self.use_tfidf:
            return
        X = self.transform(sentences)
        self.tfidf_transformer = TfidfTransformer()
        self.tfidf_transformer.fit(X)


    def transform(self, sentences):
        embedding_size = len(self.words)
        output = np.zeros(shape=(len(sentences), embedding_size))
        for index, sentence in enumerate(sentences):
            output[index] = self._transform_sentence(sentence, embedding_size)

        if self.tfidf_transformer is not None:
            return self.tfidf_transformer.transform(output).toarray()
        else:
            return output

    def _transform_sentence(self, sentence, embedding_size):
        embedding = np.zeros(shape=embedding_size)
        for token in sentence:
            if token in self.words:
                embedding[self.words[token]] += 1
        return embedding

    def dimension(self):
        return len(self.words)


def load_binary_word2vec_model(vector_path):
    if not os.path.exists(vector_path):
        raise Exception("Model doesnt exist in location %s" % vector_path)
    return gensim.models.KeyedVectors.load_word2vec_format(vector_path, binary=True, encoding='utf-8',
                                                           unicode_errors='ignore')


class CompressedEmbeddingModel:
    """Only stores vectors present in the data to save RAM. Assumes that either lowercased tokens without normalization
    or lowercased lemmas are used as input.
    """
    def __init__(self, vectors, size):
        self.vectors = vectors
        self.size = size
        self.compressed_vectors = None
        self.indices = {}

    def fit(self, sentences_text):
        unique_tokens = set()
        for sentence in sentences_text:
            # Produce both sets of tokens for each sentence
            normal_tokens = tokenizer.produce_tokens(sentence, True)
            lemmatized_tokens = tokenizer.produce_tokens(sentence, True, 'lemma')

            # Add to the set of unique tokens
            unique_tokens.update(normal_tokens)
            unique_tokens.update(lemmatized_tokens)

        # Create an array containing all the vectors needed
        self.compressed_vectors = np.zeros(shape=(len(unique_tokens), self.size))

        for index, token in enumerate(unique_tokens):
            embedding = get_token_embedding(token, self.vectors)
            # If embedding exists for token, add it to the array and add the array index to the dict
            if embedding is not None:
                self.compressed_vectors[index] = embedding
                self.indices[token] = index
        self.vectors = None

    def get_token_embedding(self, token):
        if token in self.indices:
            index = self.indices[token]
            return self.compressed_vectors[index]
        else:
            return None


class EmbeddingModel:
    """
    Maps text to numeric representation using word embeddings and an aggregation function. Assumes that the compressed
    embedding models already exist.
    """
    def __init__(self, method, model_name, model=None):
        if model_name == 'turku':
            self.model = utils.load_object(os.path.join('data', 'turku_model.json'))
        elif model_name == 'fb':
            self.model = utils.load_object(os.path.join('data', 'fb_model.json'))
        else:
            self.model = model

        self.method = method
        self.weights = {}
        self.alpha = 1e-4  # Alpha parameter for SIF
        self.pc = None

    def transform(self, sentences, skip_pc_check=False):
        output = np.zeros(shape=(len(sentences), self.model.size))
        for index, sentence in enumerate(sentences):
            embedding = self._transform_sentence(sentence)
            output[index] = embedding
        if self.pc is not None:
            return remove_pc(output, self.pc)  # Remove the 1st principal component for SIF
        elif not skip_pc_check and self.method == 'sif':
            raise AssertionError('PC not available while using SIF')
        else:
            return output

    def _transform_sentence(self, sentence):
        if self.method == 'sif' or self.method == 'avg':
            embedding = np.zeros(shape=self.model.size)
            total_weight = 0

            for token in sentence:
                token_embedding = self.model.get_token_embedding(token)
                if token_embedding is not None:
                    weight = self.get_weight(token)
                    total_weight += weight
                    embedding += token_embedding * weight
            norm = np.linalg.norm(embedding)
            if total_weight > 0 and norm != 0:
                return [x / norm for x in embedding]
            else:
                return embedding
        elif self.method == 'max':  # Take elementwise max of the vectors
            embedding = np.zeros(shape=self.model.size)
            for token in sentence:
                token_embedding = self.model.get_token_embedding(token)
                if token_embedding is not None:
                    embedding = np.maximum(embedding, token_embedding)
            return embedding
        else:
            raise ValueError('Invalid argument %s' % self.method)

    def fit(self, sentences):
        if self.method != 'sif':
            return
        # For the SIF method, calculate token counts and pre-calculate corresponding weights for the get_weight method
        frequencies = produce_token_counts(sentences, True)
        for token, freq in frequencies.items():
            self.weights[token] = self.alpha / (self.alpha + freq)

        # Pre-calculate the 1st principal component for common component removal
        X = self.transform(sentences, True)
        self.pc = compute_pc(X, 1)

    def get_weight(self, token):
        if self.method != 'sif' or not token in self.weights:
            return 1  # If using averaging method or oov token for SIF, the weight is just 1
        return self.weights[token]
