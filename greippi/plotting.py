from functools import partial
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np


def formatter(x, pos, unit):
    return '%s%s' % (x, unit)


def plot_img(path, sorted_values, unit, size, xlabel, ylabel):
    plot_formatter = partial(formatter, unit=unit)
    func_formatter = FuncFormatter(plot_formatter)

    values = []
    labels = []

    for label_value in sorted_values:
        labels.append(label_value[0])
        values.append(label_value[1])

    ticks = np.arange(len(values))
    fig, ax = plt.subplots(figsize=size)
    ax.yaxis.set_major_formatter(func_formatter)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.bar(ticks, values)
    plt.xticks(ticks, labels)
    plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment='right')
    plt.savefig(path)
    plt.close(fig)
