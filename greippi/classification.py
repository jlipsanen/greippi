import os
import warnings
import sys

from greippi import tokenizer
from greippi import utils
from greippi import visualize
from greippi import classifier
from greippi import negative_mining
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


def get_factories():
    factories = [classifier.BaselineFactory(), classifier.LogRegFactory(False), classifier.LogRegFactory(True),
                 classifier.kNNFactory(False), classifier.kNNFactory(True), classifier.MLPFactory(False),
                 classifier.MLPFactory(True), classifier.SVMFactory(False), classifier.SVMFactory(True),
                 classifier.RFFactory(False), classifier.RFFactory(True), classifier.DTFactory(False),
                 classifier.DTFactory(True), classifier.MultinomialNBFactory()]

    return factories


def cv_classifier(factory, X, Y):
    clf = factory.create_classifier()

    features, targets = shuffle(X, Y, random_state=0)
    rf = GridSearchCV(clf, param_grid=factory.get_param_grid(), n_jobs=-1, cv=10, scoring='accuracy')
    rf.fit(features, targets)
    clf.set_params(**rf.best_params_)
    print('Best params for classifier %s are %s' % (factory.name(), rf.best_params_))

    return clf, rf.best_params_


def test_dataset(X_train, Y_train, X_test, Y_test):
    factories = get_factories()
    predictions = {}
    classifiers = {}
    parameter_dict = {}
    best_score = 0
    best_predictions = []

    for factory in factories:
        print('Testing factory %s' % factory.name())
        classifier, params = cv_classifier(factory, X_train, Y_train)
        classifier.fit(X_train, Y_train)
        Y_predict = classifier.predict(X_test)
        score = accuracy_score(Y_test, Y_predict)

        if score > best_score:
            best_score = score
            best_predictions = Y_predict
        predictions[factory.name()] = Y_predict
        classifiers[factory.name()] = classifier
        parameter_dict[factory.name()] = params
        utils.save_object(os.path.join('results', '%s_labels.json' % factory.name()), Y_predict.tolist())
        print('Accuracy %.3f' % score)

    with open(os.path.join('results', 'classification.csv'), encoding='utf-8', mode='w') as file:
        file.write('Name;Accuracy mean;CI;McNemars t;Macro Accuracy;Dimension;Normalization;Stopword list;TFIDF;'
                   'Aggregation;Embedding;Params\n')
        for factory_name, Y_predict in predictions.items():
            classifier = classifiers[factory_name]
            params = parameter_dict[factory_name]

            score = accuracy_score(Y_test, Y_predict)
            ci_middle, ci = utils.get_accuracy_ci(score, len(Y_predict))
            macro_acc = utils.get_macro_acc(Y_test, Y_predict)
            mcnemars_t = utils.mcnemars_t(Y_test, Y_predict, best_predictions)

            visualize.visualize_confusion_matrix(classifier, X_test, Y_test, factory_name)
            file.write('%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n'
                       % (factory_name, ci_middle, ci, mcnemars_t, macro_acc,
                          classifier.transformer.text_N, classifier.transformer.text_normalization,
                          classifier.transformer.text_stopword, classifier.transformer.text_tfidf,
                          classifier.transformer.text_aggregation, params.get('text_embedding', None), params))


def load_test_and_train():
    train_variables = utils.load_object(os.path.join('data', 'train_variables.json'))
    train_labels = utils.load_object(os.path.join('data', 'train_labels.json'))
    test_variables = utils.load_object(os.path.join('data', 'test_variables.json'))
    test_labels = utils.load_object(os.path.join('data', 'test_labels.json'))

    train_text = tokenizer.produce_text(train_variables, ['categories', 'qstnlit', 'var_grp_qstn'])
    test_text = tokenizer.produce_text(test_variables, ['categories', 'qstnlit', 'var_grp_qstn'])

    return train_text, train_labels, test_text, test_labels


def main():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    train_text, train_labels, test_text, test_labels = load_test_and_train()
    test_dataset(train_text, train_labels, test_text, test_labels)


if __name__ == '__main__':
    main()
