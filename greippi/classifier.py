import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from greippi.tokenizer import produce_tokens
from greippi.vectorizers import BOWVectorizer, EmbeddingModel

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.svm
import sklearn.linear_model
import sklearn.neighbors
import sklearn.ensemble
import sklearn.naive_bayes


def get_nb_param_grid():
    return {
        'text_embedding': ['bow'],
        'text_normalization': [None, 'stem', 'lemma'],
        'text_stopword': [False, True],
        'text_tfidf': [False],
        'text_N': [None, 500, 250, 100, 50, 10],
        'text_aggregation': [None]
    }


def get_bow_param_grid():
    return {
        'text_embedding': ['bow'],
        'text_normalization': [None, 'stem', 'lemma'],
        'text_stopword': [False, True],
        'text_tfidf': [False, True],
        'text_N': [None, 500, 250, 100, 50, 10],
        'text_aggregation': [None]
        }


def get_embedding_param_grid():
    return {
         'text_embedding': ['turku', 'fb'],
         'text_normalization': [None, 'lemma'],
         'text_stopword': [False, True],
         'text_aggregation': ['avg', 'sif', 'max']
        }


SENT_DICT = {}


def tokenize(text, normalization, stopword):
    code = '%s - %s - %s' % (text, normalization, stopword)
    if code not in SENT_DICT:
        SENT_DICT[code] = produce_tokens(text, lowercase=True, normalization=normalization, remove_stopwords=stopword)
    return SENT_DICT[code]


def tokenize_sentences(data, normalization, stopword):
    output = []
    for sentence in data:
        output.append(tokenize(sentence, normalization, stopword))
    return output


class TextTransformer(BaseEstimator):
    def __init__(self, text_embedding='bow', text_normalization=None, text_stopword=False, text_tfidf=False,
                 text_N=None, text_aggregation=None):
        self.text_embedding = text_embedding
        self.text_normalization = text_normalization
        self.text_stopword = text_stopword
        self.text_tfidf = text_tfidf
        self.text_N = text_N
        self._vectorizer = None
        self.text_aggregation = text_aggregation

    def get_params(self, deep=True):
        return {"text_normalization": self.text_normalization,
                "text_stopword": self.text_stopword,
                "text_tfidf": self.text_tfidf,
                "text_N": self.text_N,
                "text_embedding": self.text_embedding,
                "text_aggregation": self.text_aggregation
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, data):
        sentences = tokenize_sentences(data, self.text_normalization, self.text_stopword)
        if self.text_embedding == 'bow':
            self._vectorizer = BOWVectorizer(self.text_N, 5, self.text_tfidf)
            self._vectorizer.fit(sentences)
            self.text_N = len(self._vectorizer.words)
        else:
            self._vectorizer = EmbeddingModel(self.text_aggregation, self.text_embedding)
            self._vectorizer.fit(sentences)
            self.text_N = 300
        return self

    def transform(self, data):
        sentences = tokenize_sentences(data, self.text_normalization, self.text_stopword)
        return self._vectorizer.transform(sentences)


class TextClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.transformer = TextTransformer(**kwargs)

    def fit(self, data, targets):
        self.transformer.fit(data)
        transformed = self.transformer.transform(data)
        self.classifier.fit(transformed, targets)
        self.classes_ = self.classifier.classes_
        return self

    def predict(self, data):
        transformed = self.transformer.transform(data)
        return self.classifier.predict(transformed)

    def get_params(self, deep=True):
        params1 = self.transformer.get_params()
        params2 = self.classifier.get_params()
        params1.update(params2)

        return params1

    def set_params(self, **parameters):
        classifier_params = self.classifier.get_params()
        new_classifier_params = {}
        new_transformer_params = {}

        for parameter, value in parameters.items():
            if parameter in classifier_params:
                new_classifier_params[parameter] = value
            else:
                new_transformer_params[parameter] = value
        self.classifier.set_params(**new_classifier_params)
        self.transformer.set_params(**new_transformer_params)
        return self


def split_params(kwargs, classifier_params):
    new_classifier_params = {}
    super_params = {}
    for parameter, value in kwargs.items():
        if parameter in classifier_params:
            new_classifier_params[parameter] = value
        else:
            super_params[parameter] = value
    return new_classifier_params, super_params


class kNNClassifier(TextClassifier):
    def __init__(self, **kwargs):
        classifier_params, super_params = split_params(kwargs, sklearn.neighbors.KNeighborsClassifier().get_params())
        self.classifier = sklearn.neighbors.KNeighborsClassifier(**classifier_params)
        TextClassifier.__init__(self, **super_params)


class NBClassifier(TextClassifier):
    def __init__(self, **kwargs):
        classifier_params, super_params = split_params(kwargs, sklearn.naive_bayes.MultinomialNB().get_params())
        self.classifier = sklearn.naive_bayes.MultinomialNB(**classifier_params)
        TextClassifier.__init__(self, **super_params)


class LogisticClassifier(TextClassifier):
    def __init__(self, **kwargs):
        classifier_params, super_params = split_params(kwargs, sklearn.linear_model.LogisticRegression().get_params())
        self.classifier = sklearn.linear_model.LogisticRegression(**classifier_params)
        TextClassifier.__init__(self, **super_params)


class SVMClassifier(TextClassifier):
    def __init__(self, **kwargs):
        classifier_params, super_params = split_params(kwargs, sklearn.svm.SVC().get_params())
        self.classifier = sklearn.svm.SVC(**classifier_params)
        TextClassifier.__init__(self, **super_params)


class RFClassifier(TextClassifier):
    def __init__(self, **kwargs):
        classifier_params, super_params = split_params(kwargs, sklearn.ensemble.RandomForestClassifier().get_params())
        self.classifier = sklearn.ensemble.RandomForestClassifier(**classifier_params)
        TextClassifier.__init__(self, **super_params)


class DTClassifier(TextClassifier):
    def __init__(self, **kwargs):
        classifier_params, super_params = split_params(kwargs, DecisionTreeClassifier().get_params())
        self.classifier = DecisionTreeClassifier(**classifier_params)
        TextClassifier.__init__(self, **super_params)


class NNClassifier(TextClassifier):
    def __init__(self, **kwargs):
        classifier_params, super_params = split_params(kwargs, MLPClassifier().get_params())
        self.classifier = MLPClassifier(**classifier_params)
        TextClassifier.__init__(self, **super_params)


class BaselineClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self._predicted_class = ''
        self.transformer = TextTransformer()
        self.classes_ = None

    def fit(self, data, targets):
        unique_classes = set(targets)
        max_count = 0
        self.classes_ = list(unique_classes)

        for unique_class in unique_classes:
            count = targets.count(unique_class)
            if count > max_count:
                max_count = count
                self._predicted_class = unique_class
        return self

    def predict(self, data):
        return np.array([self._predicted_class] * len(data))

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        return self


class kNNFactory:
    def __init__(self, bow):
        self.bow = bow

    def get_param_grid(self):
        params = {'n_neighbors': [1, 2, 5, 10, 20], 'metric': ['euclidean', 'cosine']}
        if self.bow:
            grid = get_bow_param_grid()
        else:
            grid = get_embedding_param_grid()
        params.update(grid)
        return params

    def name(self):
        return "k-NN-%s" % self.bow

    def create_classifier(self):
        return kNNClassifier()


class MultinomialNBFactory:
    def get_param_grid(self):
        params = {'alpha': [1e-9, 1e-6, 1e-3, 1, 10, 100]}
        params.update(get_nb_param_grid())
        return params

    def name(self):
        return "multinb"

    def create_classifier(self):
        return NBClassifier()


class LogRegFactory:
    def __init__(self, bow):
        self.bow = bow

    def name(self):
        return "logreg-%s" % self.bow

    def get_param_grid(self):
        params = {'C': [0.1, 1, 10, 100, 1000], 'multi_class': ['ovr'], 'solver': ['liblinear'], 'random_state': [0]}
        if self.bow:
            grid = get_bow_param_grid()
        else:
            grid = get_embedding_param_grid()
        params.update(grid)
        return params

    def create_classifier(self):
        return LogisticClassifier()


class SVMFactory:
    def __init__(self, bow):
        self.bow = bow

    def name(self):
        return "svm-%s" % self.bow

    def get_param_grid(self):
        params = {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'linear'], 'gamma': ['scale'], 'cache_size': [2000],
                  'max_iter': [1000], 'random_state': [0]}
        if self.bow:
            grid = get_bow_param_grid()
        else:
            grid = get_embedding_param_grid()
        params.update(grid)
        return params

    def create_classifier(self):
        return SVMClassifier()


class RFFactory:
    def __init__(self, bow):
        self.bow = bow

    def get_param_grid(self):
        params = {'n_estimators': [50, 100, 200, 400], 'random_state': [0]}
        if self.bow:
            grid = get_bow_param_grid()
        else:
            grid = get_embedding_param_grid()
        params.update(grid)
        return params

    def name(self):
        return "rf-%s" % self.bow

    def create_classifier(self):
        return RFClassifier()


class DTFactory:
    def __init__(self, bow):
        self.bow = bow

    def get_param_grid(self):
        params = {"criterion": ["gini", "entropy"], "max_depth": [16, 32, 64, 128, 256, 512, 1024], 'random_state': [0]}
        if self.bow:
            grid = get_bow_param_grid()
        else:
            grid = get_embedding_param_grid()
        params.update(grid)
        return params

    def name(self):
        return "dt-%s" % self.bow

    def create_classifier(self):
        return DTClassifier()


class MLPFactory:
    def __init__(self, bow):
        self.bow = bow

    def get_param_grid(self):
        params = {"hidden_layer_sizes": [(200, 100), (100, 50), (50, 25)], 'random_state': [0]}
        if self.bow:
            grid = get_bow_param_grid()
        else:
            grid = get_embedding_param_grid()
        params.update(grid)
        return params

    def name(self):
        return "mlp-%s" % self.bow

    def create_classifier(self):
        return NNClassifier()


class BaselineFactory:
    def name(self):
        return "baseline"

    def get_param_grid(self):
        return {}

    def create_classifier(self):
        return BaselineClassifier()
