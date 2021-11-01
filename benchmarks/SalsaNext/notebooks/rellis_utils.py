import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm


def get_config(config_file=None):
    """
    Get the configurations of the Rellis-3D dataset
    """
    DEFAULT_CONFIG_FILE = "../train/tasks/semantic/config/labels/rellis.yaml"
    if config_file is None:
        config_file = DEFAULT_CONFIG_FILE
    rellis_config = yaml.safe_load(open(config_file, "r"))
    return rellis_config


def get_label_to_cid_map(config):
    """
    Get the label vector from a label file
    """
    label_to_cid_map = config["learning_map"]
    return label_to_cid_map


def get_label_vector(filepath):
    """
    Get the label vector from a label file
    """
    return np.fromfile(filepath, dtype=np.int32).reshape((-1))


def labels_to_cids(labels, label_to_cid_map):
    """
    Convert a label vector to a class id vector
    """
    tmp = labels.copy()
    for k, v in label_to_cid_map.items():
        labels[tmp == k] = v
    return labels


def get_classnames(rellis_config):
    """
    Get the label vector from a label file
    """
    name_map = dict()
    for cid, label in rellis_config["learning_map_inv"].items():
        name_map[cid] = rellis_config["labels"][label]
    classnames = [name_map[i] for i in range(len(name_map))]
    return classnames


def calc_confusion_matrix(g, p, n, ignore_class_ids):
    """
    Get confusion matrix given a ground truth vector and a prediction vector

    g -- ground truth vector
    p -- prediction vector
    n -- number of classes
    ignore -- ignore class ids
    """
    for cid in ignore_class_ids:
        g = g[g != cid]
        p = p[p != cid]

    # compare[i] = g[i] * n + p[i]: prediction for
    # point i is pred[i] where ground truth is labels[i]
    compare = g * n + p

    # bins[g[i] * n + p[j]]: number of points with
    #     ground-truth g[i] and are predicted to be p[j]
    bins = np.bincount(compare)
    # pad with zeros
    bins = np.pad(bins, (0, n * n - len(bins)), "constant")
    assert bins.shape[0] == n * n

    cm = np.zeros((n, n), dtype=np.int32)
    for gcid in range(n):
        for pcid in range(n):
            cm[gcid, pcid] = bins[gcid * n + pcid]
    return cm


from os import listdir
from os.path import isfile, join


def get_sequence_frequency(seq, data_dir):
    assert seq in ["00000", "00001", "00002", "00003", "00004"]
    seq_dir = f"{data_dir}/{seq}/os1_cloud_node_semantickitti_label_id/"
    files = [f for f in listdir(seq_dir) if isfile(join(seq_dir, f))]

    classnames = get_classnames()

    n = len(classnames)
    count = np.zeros(n, dtype=np.int32)

    for fname in tqdm(files):
        glabels = get_label_vector(f"{seq_dir}/{fname}")
        g = labels_to_cids(glabels)
        bins = np.bincount(g)
        bins = np.pad(bins, (0, n - len(bins)), "constant")
        count = count + bins

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 2, 2])
    ax.bar(classnames, count)
    ax.set_xlabel("Semantic class")
    ax.set_ylabel("Fequency (number of points)")
    ax.set_title(f"Frequency of all semantic classes - Sequence {seq}")
    for index, value in enumerate(count):
        ax.text(index, value, str(count[index]), ha="center")
    plt.show()

    return count


def get_split_frequency(split_file, data_dir, desc=None, color=None):
    label_filenames = []

    with open(split_file, "r") as f:
        for line in f:
            label_filename = line.strip().split(" ")[1]
            label_filenames.append(label_filename)

    config = get_config()
    classnames = get_classnames(config)
    label_to_cid_map = get_label_to_cid_map(config)
    n = len(classnames)
    count = np.zeros(n, dtype=np.int32)

    for fname in tqdm(label_filenames):
        glabels = get_label_vector(f"{data_dir}/{fname}")
        g = labels_to_cids(glabels, label_to_cid_map)
        bins = np.bincount(g)
        bins = np.pad(bins, (0, n - len(bins)), "constant")
        count = count + bins

    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 2, 2])
    # ax.bar(classnames, count, color=color)
    # ax.set_xlabel("Semantic class")
    # ax.set_ylabel("Fequency (number of points)")
    # ax.set_title(f"Frequency of all semantic classes\n{desc}")
    # for index, value in enumerate(count):
    #     ax.text(index, value, str(count[index]), ha="center")
    # plt.show()

    return count


class Evaluator:
    def __init__(self, data_dir, pred_dir, desc, config_file=None):
        self.data_dir = data_dir
        self.pred_dir = pred_dir
        self.desc = desc
        self.config = get_config(config_file=config_file)
        self.classnames = get_classnames(self.config)
        self.label_to_cid_map = get_label_to_cid_map(self.config)
        self.iou = None
        self.mean_iou = None

    def calc_iou(self):
        label_filenames = []

        with open(f"{self.data_dir}/pt_test.lst", "r") as f:
            for line in f:
                label_filename = line.strip().split(" ")[1]
                label_filenames.append(label_filename)

        n = len(self.classnames)
        cm = np.zeros((n, n), dtype=np.int32)

        for fname in tqdm(label_filenames):
            glabels = get_label_vector(f"{self.data_dir}/{fname}")
            plabels = get_label_vector(f"{self.pred_dir}/{fname}")
            g = labels_to_cids(glabels, self.label_to_cid_map)
            p = labels_to_cids(plabels, self.label_to_cid_map)
            cm += calc_confusion_matrix(g, p, n, [])

        iou = np.diag(cm) / np.maximum(
            1.0, cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)
        )
        self.iou = iou
        self.mean_iou = iou.mean()
        return self.iou, self.mean_iou

    def plot_iou(self):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1.5, 1.5])
        ax.bar(self.classnames, self.iou)
        ax.set_xlabel("Semantic classes")
        ax.set_ylabel("IoU")
        ax.set_title(f"IoU of all semantic classes\n{self.desc}")
        plt.show()
