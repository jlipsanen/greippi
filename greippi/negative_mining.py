import os
import json
import glob
from greippi import utils
from greippi import visualize


def load_labels():
    label_dict = dict()
    for filepath in glob.glob(os.path.join('results', '*')) + glob.glob(os.path.join('data', '*')):
        if filepath.endswith('_labels.json'):
            filename = os.path.basename(filepath)
            index = filename.index('_labels.json')
            label_name = filename[0:index]
            print(label_name)
            if label_name != 'train' and label_name != 'baseline':
                label_dict[label_name] = utils.load_object(filepath)
    return label_dict

def load_variables():
    return utils.load_object(os.path.join('data', 'test_variables.json'))

def get_correct(label_dict, variables):
    ground_truth = label_dict['test']
    index_incorrect = {}

    for index in range(len(ground_truth)):
        index_incorrect[index] = 0

    total_incorrect = 0

    for key, labels in label_dict.items():
        if key not in ['test', 'train']:
            correct = 0
            total = 0
            for i, label in enumerate(labels):
                if ground_truth[i] == label:
                    correct += 1
                else:
                    index_incorrect[i] += 1
                    total_incorrect += 1
                total += 1
            print(key, correct / total, "accuracy")
    print("total variables", total)
    print("total variables with incorrect classification", len([index for index, count in index_incorrect.items() if
                                                                count > 0]))
    inverse_sort = sorted(index_incorrect.items(), key=lambda x: -x[1])
    wrong_list = []
    top100_incorrect = 0
    visualize.draw_barchart([x[1] for x in inverse_sort])

    for index, value in inverse_sort[0: 100]:
        obj = dict()
        obj['variable'] = variables[index]
        obj['truth'] = ground_truth[index]
        obj['wrong'] = {}
        obj['right'] = []
        for key, labels in label_dict.items():
            if key not in ['test', 'train']:
                if ground_truth[index] != labels[index]:
                    obj['wrong'][key] = labels[index]
                    top100_incorrect += 1
                else:
                    obj['right'].append(key)
        wrong_list.append(obj)
    print("Top 100 incorrect had", top100_incorrect / total_incorrect, "of the incorrectly classified variables")
    pretty_printed = json.dumps(wrong_list, indent=4, sort_keys=True, ensure_ascii=False)
    with open(os.path.join('results', 'negative_mining.json'), mode='w', encoding='utf-8') as fp:
        fp.write(pretty_printed)


def mine_negative_examples():
    label_dict = load_labels()
    variables = load_variables()
    get_correct(label_dict, variables)


if __name__ == '__main__':
    mine_negative_examples()