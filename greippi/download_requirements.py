import nltk
nltk.download('stopwords')
nltk.download('punkt')

import os
import fasttext
from greippi.vectorizers import CompressedEmbeddingModel, load_binary_word2vec_model
from greippi import utils
from greippi import classification
from greippi import tokenizer


def init_compressed_embedding_models(sentences_text):
    if not os.path.exists(os.path.join('data', 'fb_model.json')):
        fasttext.util.download_model('fi', if_exists='ignore')
        facebook_vectors = fasttext.load_model('cc.fi.300.bin')
        _FACEBOOK_MODEL = CompressedEmbeddingModel(facebook_vectors, 300)
        _FACEBOOK_MODEL.fit(sentences_text)
        utils.save_object(os.path.join('data', 'fb_model.json'), _FACEBOOK_MODEL)
        facebook_vectors = None

    if not os.path.exists(os.path.join('data', 'turku_model.json')):
        embeddings_path = 'fin-word2vec.bin'
        if not os.path.exists(embeddings_path):
            print('turkunlp vectors not available, downloading...')
            utils.download_file('http://bionlp-www.utu.fi/fin-vector-space-models/fin-word2vec.bin', embeddings_path)
        turku_vectors = load_binary_word2vec_model(embeddings_path)
        _TURKU_MODEL = CompressedEmbeddingModel(turku_vectors, 300)
        _TURKU_MODEL.fit(sentences_text)
        utils.save_object(os.path.join('data', 'turku_model.json'), _TURKU_MODEL)
        _TURKU_MODEL = None


def main():
    tokenizer.init_voikko()
    train_text, train_labels, test_text, test_labels = classification.load_test_and_train()
    init_compressed_embedding_models(train_text + test_text)


if __name__ == '__main__':
    main()