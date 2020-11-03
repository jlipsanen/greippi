import math

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

import requests
import clint.textui


def save_object(filepath, object):
    with open(filepath, mode='w', encoding='utf-8') as file:
        encoded = jsonpickle.encode(object)
        file.write(encoded)


def load_object(filepath):
    with open(filepath, mode='r', encoding='utf-8') as file:
        return jsonpickle.decode(file.read())


def produce_token_counts(sentences, calculate_frequencies=False):
    counts = {}
    total = 0
    for sentence in sentences:
        for token in sentence:
            if token not in counts:
                counts[token] = 0
            counts[token] += 1
            total += 1
    if calculate_frequencies:
        for token in counts:
            counts[token] /= total
    return counts


def mcnemars_t(truth, predicted1, predicted2):
    n_01 = 0
    n_10 = 0
    for true_label, label1, label2 in zip(truth, predicted1, predicted2):
        if true_label == label1 and true_label != label2:
            n_10 += 1
        elif true_label != label1 and true_label == label2:
            n_01 += 1
    divisor = (n_01 + n_10)
    if divisor == 0:
        return float('-inf')
    t = ((abs(n_01 - n_10) - 1) ** 2) / divisor
    return round(t, 3)


def get_accuracy_ci(accuracy, n):
    z = 1.96
    t = z ** 2 / n
    est = (accuracy + t / 2) / (1 + t)
    plusminus = math.sqrt(accuracy * (1 - accuracy) * t + t**2 / 4) / (1 + t)
    return round(est, 3), round(plusminus, 3)


def get_macro_acc(truth, predicted):
    labels = set(truth)
    avg = 0
    for label in labels:
        count = 0
        correct = 0
        for index, true_label in enumerate(truth):
            if true_label == label:
                test_label = predicted[index]
                count += 1
                if test_label == true_label:
                    correct += 1
        avg += float(correct) / count

    avg /= len(labels)
    return round(avg, 3)


def download_file(url, filepath):
    request = requests.get(url, stream=True)
    with open(filepath, 'wb') as f:
        total_length = int(request.headers.get('content-length'))
        for chunk in clint.textui.progress.bar(request.iter_content(chunk_size=1024),
                                               expected_size=(total_length / 1024) + 1):
            if chunk:
                f.write(chunk)
                f.flush()