import os
import zipfile

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import libvoikko

from greippi import utils

_STEMMER = SnowballStemmer(language='finnish')
_STOPWORDS = set(stopwords.words('finnish'))


def lowercase_tokens(tokens):
    return [token.lower() for token in tokens]


def stem_tokens(tokens):
    return [_STEMMER.stem(token) for token in tokens]


def remove_stopwords_tokens(tokens):
    return [token for token in tokens if token.lower() not in _STOPWORDS]


def extract_text(variable, fields):
    text = ''

    for field in fields:
        if field not in variable:
            continue
        value = variable[field]
        if text:
            text += ' '
        if isinstance(value, list):
            text += ' '.join(value)
        else:
            text += value

    return text


def tokenize(text, lowercase=False, remove_stopwords=False, stem=False, lemmatize=False):
    tokenizer = VoikkoTokenizer()
    tokens = tokenizer.get_tokens(text, lemmatize)

    if remove_stopwords:
        tokens = remove_stopwords_tokens(tokens)

    if lowercase:
        tokens = lowercase_tokens(tokens)

    if stem:
        tokens = stem_tokens(tokens)

    return tokens


def produce_tokens(text, lowercase=False, normalization=None, remove_stopwords=False):
    if normalization == 'stem':
        stem = True
        lemmatize = False
    elif normalization == 'lemma':
        stem = False
        lemmatize = True
    else:
        stem = False
        lemmatize = False

    return tokenize(text, lowercase=lowercase, stem=stem, remove_stopwords=remove_stopwords, lemmatize=lemmatize)


def produce_text(variables, fields):
    print("Producing text for %s variables, with fields %s" % (len(variables), fields))
    text_list = []
    for variable in variables:
        text_list.append(extract_text(variable, fields))
    return text_list


DICTIONARY_PATH = os.path.join('data', 'dictionary')
VOIKKO_DICT_PATH = os.path.join('data', 'dict-morphoid.zip')
_voikko = None


if os.path.exists(DICTIONARY_PATH):
    _voikko = _voikko = libvoikko.Voikko('fi', DICTIONARY_PATH)
else:
    print('Voikko dictionary missing!')


def download_voikko_dictionary():
    print('Missing voikko dictionary, downloading...')
    utils.download_file('https://www.puimula.org/htp/testing/voikko-snapshot-v5/dict-morphoid.zip',
                   VOIKKO_DICT_PATH)


def extract_voikko_dictionary():
    print('Extracting voikko dictionary to path...')
    global _voikko
    with zipfile.ZipFile(VOIKKO_DICT_PATH, 'r') as zip:
        zip.extractall(DICTIONARY_PATH)
    _voikko = libvoikko.Voikko('fi', DICTIONARY_PATH)


def init_voikko():
    if not os.path.exists(VOIKKO_DICT_PATH):
        download_voikko_dictionary()
    if not os.path.exists(DICTIONARY_PATH):
        extract_voikko_dictionary()


class VoikkoTokenizer:
    def get_tokens(self, text, lemmatize):
        tokens = _voikko.tokens(text)
        output = []

        for token in tokens:
            if token.tokenType == token.WORD:
                if lemmatize:
                    tokens = self.lemmatize_word(token.tokenText)
                    output.extend(tokens)
                else:
                    output.append(token.tokenText)
        return output

    def lemmatize_word(self, word):
        analyzed = _voikko.analyze(word)
        if analyzed:
            return set([word['BASEFORM'].lower() for word in analyzed])
        else:
            return word
