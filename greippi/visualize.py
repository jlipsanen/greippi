from functools import partial
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import plot_confusion_matrix

def visualize_confusion_matrix(classifier, X, Y, classifier_name):
    MAX_LENGTH = 14
    labels = sorted(np.unique(Y))
    def shorten(index, pos, labels):
        label = labels[index]
        if len(label) > MAX_LENGTH:
            return label[0:7] + '...'
        return label

    fig, ax = plt.subplots(figsize=(14, 12))
    plot_confusion_matrix(classifier, X, Y, xticks_rotation='vertical', ax=ax, cmap='Greys')
    plt.xlabel('Ennustettu luokka')
    plt.ylabel('Oikea luokka')
    ax.xaxis.set_major_formatter(FuncFormatter(partial(shorten, labels=labels)))
    ax.yaxis.set_major_formatter(FuncFormatter(partial(shorten, labels=labels)))

    figures_path = os.path.join('results', 'figures')
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    plt.savefig(os.path.join(figures_path, '%s.png' % classifier_name))
    plt.savefig(os.path.join(figures_path, '%s.svg' % classifier_name))
    plt.close()

def draw_barchart(bars):
    fig, ax = plt.subplots(figsize=(14, 12))
    indices = np.arange(len(bars))

    p1 = plt.bar(indices, bars)
    plt.ylabel('Väärin')
    plt.xlabel('Muuttujan numero')
    plt.show()

    figures_path = os.path.join('results', 'figures')
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    plt.savefig(os.path.join(figures_path, 'error_barchart.png'))
    plt.savefig(os.path.join(figures_path, 'error_barchart.svg'))
    plt.close()
