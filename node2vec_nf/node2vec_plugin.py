# -*- coding: utf-8 -*-
#
# Copyright 2017-2018 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import sys
import os
import time
import networkx as nx
import numpy as np

from utils.epgm import EPGM
from gensim.models import Word2Vec
from utils.node2vec import node2vec
from distutils.util import strtobool
import multiprocessing

import logging
logger = logging.getLogger('sciluigi-interface')


def Node2Vec_task(worker, inputs={}, output=None, parameters={}):  # this is the executor, i.e., the actual function executing the task
    """
    Learn Node2Vec embedding for graph vertex representation learning.

    Args:
        worker (obj): The server handler.
        inputs (dict): Input data.

            - "graph_filename": <str>
            - "dataset_name": <str>

        output (str): Output directory name
        parameters (dict): All parameters specific to node2vec algorithm.

            - "p":  <float>,
            - "q": <float>,
            - "dimensions": <int>,
            - "numWalks": <int>,
            - "walkLength": <int>,
            - "window": <int>,
            - "iter": <int>,
            - "workers": <int>,
            - "weighted": <True/False>,
            - "directed": <True/False>,

    Notes:
        Reasonable default parameter values are already supplied in
        the plugin so if none are given, it still works.

    Returns:
        Saves the representations as a .emb file and returns it's location
        as 'embeddings_filename'
    """
    default_params = {
        'p': 1.,  # Parameter p
        'q': 1.,  # Parameter q
        'dimensions': 128,  # dimensionality of node2vec embeddings
        'numWalks': 10,  # Number of walks from each node
        'walkLength': 80,  # Walk length
        'window': 10,  # Context size for word2vec
        'iter': 1,  # number of SGD iterations (epochs)
        'workers': multiprocessing.cpu_count(),  # number of workers for word2vec
        'weighted': False,  # is graph weighted?
        'directed': False  # are edges directed?
    }

    data_spec = inputs['in_data']  # was in_vector
    graph_file = os.path.expanduser(data_spec['nx_filename'])  # was edge_list_filename
    dataset_name = data_spec.get('dataset_name', None)

    # just make sure that node2vec parameters have some default values even if these have not been specified
    # in the POST request
    parameters['p'] = float(parameters.get('p', default_params['p']))
    parameters['q']= float(parameters.get('q', default_params['q']))
    parameters['dimensions'] = int(parameters.get('dimensions', default_params['dimensions']))
    parameters['num_walks'] = int(parameters.get('numWalks', default_params['numWalks']))
    parameters['walk_length'] = int(parameters.get('walkLength', default_params['walkLength']))
    parameters['window_size'] = int(parameters.get('window', default_params['window']))
    parameters['iter'] = int(parameters.get('iter', default_params['iter']))
    parameters['workers'] = int(parameters.get('workers', default_params['workers']))
    parameters['weighted'] = parameters.get('weighted', default_params['weighted'])
    parameters['directed'] = parameters.get('directed', default_params['directed'])

    if parameters['num_walks'] <= 0:
        raise Exception('number of walks per node {} specified; should be positive! Aborting.'.format(parameters['num_walks']))
    if parameters['walk_length'] <= 0:
        raise Exception('walk length of {} specified; should be positive! Aborting.'.format(parameters['walk_length']))
    if parameters['window_size'] <= 0:
        raise Exception('window size of {} specified; should be positive! Aborting.'.format(parameters['window_size']))
    if parameters['iter'] < 0:
        raise Exception('{} sgd iterations specified; should be non-negative! Aborting.'.format(parameters['iter']))

    logger.info("Loading the graph from {}...".format(graph_file))
    nx_G = worker.load_pickle(graph_file)

    logger.info("Preprocessing the graph...")
    G = node2vec.Graph(nx_G, parameters['directed'], parameters['p'], parameters['q'])

    logger.info("Preprocessing transition probabilities...")
    G.preprocess_transition_probs()

    logger.info("Simulating random walks ({} walks per node)...".format(parameters['num_walks']))

    startTime_ = time.time()
    walks = G.simulate_walks(parameters['num_walks'], parameters['walk_length'])
    endTime_ = time.time()

    logger.info("---- Random walks generation completed in "
                 + str(round(endTime_ - startTime_, 3)) + " seconds; average of "
                 + str(round((endTime_ - startTime_) / parameters['num_walks'], 3))
                 + " seconds per set of walks")

    logger.info("Learning node embedding function for {} SGD iterations...".format(parameters['iter']))

    sentences = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(sentences,
                     size=parameters["dimensions"],
                     window=parameters["window_size"],
                     min_count=0,
                     sample=0,
                     sg=1,
                     negative=16,
                     workers=parameters["workers"],
                     iter=parameters["iter"])

    # Reindex embeddings to numerical vertex value
    embeddings = np.zeros(model.wv.vectors.shape, dtype=np.float32)
    count = np.zeros(model.wv.vectors.shape[0], dtype='int')
    for word in model.wv.vocab:
        vertex = int(word)
        embeddings[vertex] = model.wv.get_vector(word)
        count[vertex] += 1

    assert all(count == 1)

    output_filename = worker.save_as_pickle(embeddings, "embeddings.pkl")
    model_filename = worker.save_as_pickle(model.wv, "w2v_vectors.pkl")
    worker.save_as_pickle(count, "w2v_count.pkl")

    task_results = {
        'rl_args': parameters,
        'embeddings_filename': output_filename,
        'model_filename': model_filename
    }

    return task_results


SERVER_TASKS = {"representation_learning": Node2Vec_task}
