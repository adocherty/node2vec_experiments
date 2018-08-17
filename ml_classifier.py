import os
import pickle
import errno

import numpy as np
import pandas as pd
import six
import tensorflow as tf
from sklearn.dummy import DummyClassifier
from sklearn import (model_selection, linear_model, multiclass,
                     preprocessing)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('train_split', 0.4, 'initial learning rate.')
flags.DEFINE_float('learning_rate', 0.2, 'initial learning rate.')
flags.DEFINE_string('train_prefix', '',
                    'name of the object file that stores the training data. must be specified.')
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_string('input_dir', '.', 'Input data directory.')
flags.DEFINE_string('emb_file', None, 'Input label file name.')
flags.DEFINE_string('label_file', None, 'Input label file name.')
flags.DEFINE_string('degrees_file', None, 'Input node degrees file name.')
flags.DEFINE_string('bscc_file', None,
                    'Biggest Strongly Connected Component file')
flags.DEFINE_string('label_dir', None, 'Input label file directory.')
flags.DEFINE_string('degrees_dir', None, 'Input node degrees file directory.')
flags.DEFINE_integer('output_index', 0, 'Output file suffix.')
flags.DEFINE_string('delimiter', '\t', 'Delimiter.')
flags.DEFINE_integer('seed', 58125312, "Seed for random generator.")
flags.DEFINE_integer('force_offset', 0, "Offset to adjust node IDs.")
flags.DEFINE_boolean('random_classifier', False,
                     'To use random classifier.')


def eval_classification(labels, embeddings, use_ml_splitter=False):
    # Classifier choice
    if FLAGS.random_classifier:
        classifier = DummyClassifier(random_state=FLAGS.seed)
    else:
        classifier = linear_model.LogisticRegression(C=10, random_state=FLAGS.seed)
        # classifier = svm.SVC(C=1)

        # Use multi-class/multi-label classifier
        # Note: for two classes this gracefully falls
        # back to binary classification.
        classifier = multiclass.OneVsRestClassifier(classifier)

    # Choose multi-label or multi-class classification
    # based on label size: we can't use StratifiedShuffleSplit
    # for the mutli-label case
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        print("Performing multi-label classification")
        shuffle = model_selection.ShuffleSplit(n_splits=10, train_size=FLAGS.train_split,
                                               test_size=1.0 - FLAGS.train_split)

    else:
        shuffle = model_selection.StratifiedShuffleSplit(
            n_splits=10, train_size=FLAGS.train_split, test_size=1.0 - FLAGS.train_split)

    scoring = ['accuracy', 'f1_macro', 'f1_micro']

    cv_scores = model_selection.cross_validate(
        classifier, embeddings, labels, scoring=scoring, cv=shuffle,
        return_train_score=True
    )
    train_acc = cv_scores['train_accuracy'].mean()
    train_f1 = cv_scores['train_f1_macro'].mean()
    test_acc = cv_scores['test_accuracy'].mean()
    test_f1 = cv_scores['test_f1_macro'].mean()

    print("Train acc: {:0.4f}, f1: {:0.4f}".format(train_acc, train_f1))
    print("Test acc: {:0.4f}, f1: {:0.4f}".format(test_acc, test_f1))

    return {'train_acc': train_acc, 'test_acc': test_acc, 'train_f1': train_f1, 'test_f1': test_f1}


def load_labels(label_filename, vocab_size):
    """Load labels file. Supports single or multiple labels"""
    raw_labels = {}
    min_labels = np.inf
    max_labels = 0
    with open(label_filename) as f:
        for line in f.readlines():
            values = [int(x) for x in line.strip().split()]
            raw_labels[values[0]] = values[1:]
            min_labels = min(len(values) - 1, min_labels)
            max_labels = max(len(values) - 1, max_labels)
    print("Raw Labels: {}".format(len(raw_labels)))
    if min_labels < 1:
        raise RuntimeError("Expected 1 or more labels in file {}"
                           .format(label_filename))
    # Single label
    elif max_labels == 1:
        labels = np.full(vocab_size, np.nan, dtype=np.int32)
        for (index, label) in six.iteritems(raw_labels):
            labels[index + FLAGS.force_offset] = label[0]
        return raw_labels, labels

    # Multiple labels
    else:
        print("Multi-label classification")
        unique_labels = np.unique(
            [l for labs in raw_labels.values() for l in labs])
        n_labels = len(unique_labels)
        print("Number of labels: {}".format(n_labels))

        label_encoder = preprocessing.MultiLabelBinarizer(unique_labels)
        labels = np.zeros((vocab_size, n_labels), dtype=np.int8)
        for (index, multi_label) in six.iteritems(raw_labels):
            labels[index + FLAGS.force_offset] = \
                label_encoder.fit_transform([multi_label])
        return raw_labels, labels


def read_existing_vocab(degree_file):
    degrees = pd.read_csv(degree_file, delimiter=FLAGS.delimiter, dtype='int32', header=None).values
    vocabs = degrees[:, 0] + FLAGS.force_offset
    return vocabs


def read_vocabs(vocab_file):
    vocabs = pd.read_csv(vocab_file, delimiter=FLAGS.delimiter, dtype='int32', header=None).values
    return vocabs[:, 0] + FLAGS.force_offset


def save_scores(scores, prefix):
    if not os.path.exists(os.path.dirname(FLAGS.base_log_dir + "/")):
        try:
            os.makedirs(os.path.dirname(FLAGS.base_log_dir + "/"))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(os.path.join(FLAGS.base_log_dir, prefix + "{}.txt".format(FLAGS.output_index)),
              "a") as f:
        f.write("{:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}\n"
                .format(scores["train_acc"], scores["train_f1"], scores["test_acc"],
                        scores["test_f1"]))


if __name__ == "__main__":
    if FLAGS.delimiter == r'\t':
        print("TAB separated")
        FLAGS.delimiter = "\t"

    print("Force Offset: {}".format(FLAGS.force_offset))
    all_embeddings = None
    with open(os.path.join(FLAGS.input_dir, FLAGS.emb_file), "rb") as pfile:
        all_embeddings = pickle.load(pfile)
    v_size = len(all_embeddings)
    print("Vocab size: {}".format(v_size))
    r_labels, all_labels = load_labels(os.path.join(FLAGS.label_dir, FLAGS.label_file), v_size)
    un, counts = np.unique(all_labels, return_counts=True)
    print(dict(zip(un, counts)))
    print("Labeled vocab size: {}".format(len(all_labels)))

    if FLAGS.bscc_file is not None:
        bscc_graph_vocabs = read_vocabs(
            os.path.join(FLAGS.degrees_dir, FLAGS.bscc_file))
        print("BSCC graph vocab size: {}".format(len(bscc_graph_vocabs)))
        bscc_evals = eval_classification(all_labels[bscc_graph_vocabs],
                                         all_embeddings[bscc_graph_vocabs])
        save_scores(bscc_evals, "bscc-scores")
    else:
        existing_vocab = read_existing_vocab(os.path.join(FLAGS.degrees_dir, FLAGS.degrees_file))
        print("Existing vocab size: {}".format(len(existing_vocab)))
        evals = eval_classification(all_labels[existing_vocab], all_embeddings[existing_vocab])
        save_scores(evals, "scores")
