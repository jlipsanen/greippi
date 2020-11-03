import os

from greippi.plotting import plot_img
from greippi import utils


def produce_statistics():
    train_labels = utils.load_object(os.path.join('data', 'train_labels.json'))
    produce_statistics_from_labels(train_labels, os.path.join('results', 'train'))


def produce_bar_chart(txt_file, img_path, value_dict, size, xlabel, ylabel, sort_desc=True, topn=None, unit=''):
    if sort_desc:
        sorted_items = sorted(value_dict.items(), key=lambda x: -x[1])
    else:
        sorted_items = sorted(value_dict.items(), key=lambda x: x[1])

    if topn is not None:
        sorted_items = sorted_items[:topn:]

    for item in sorted_items:
        txt_file.write('%s: %s\n' % (item[0], item[1]))

    plot_img(img_path, sorted_items, unit, size, xlabel, ylabel)


def produce_statistics_from_labels(labels, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.isdir(output_folder):
        print('Output folder name is already a file!')
        return

    label_set = set(labels)
    with open(os.path.join(output_folder, 'labels.txt'), encoding='utf-8', mode='w') as file:
        file.write('Number of labels: %s\n' % len(labels))
        file.write('Number of unique labels: %s\n' % len(label_set))

        values = {}
        for label in label_set:
            relevant = [var_label for var_label in labels if var_label == label]
            values[label] = len(relevant)
        produce_bar_chart(file, os.path.join(output_folder, 'labels.png'), values, (12, 10), 'Luokat', 'Lukumäärä')


if __name__ == '__main__':
    produce_statistics()
