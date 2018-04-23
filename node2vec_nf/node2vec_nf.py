# -*- coding: utf-8 -*-
#
# This file is part of stellar-evaluation, the evaluation framework
# of the Stellar project.
#
# Copyright 2017-2018 CSIRO Data61
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

import argparse
import os
import logging
import sciluigi as sl
import stellar_evaluation as se

from preprocess_plugin import preprocess_nf
from plugins.representation_learning_plugin import Node2Vec_task

# Use sciluigi logger
logger = logging.getLogger('sciluigi-interface')

# ------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------
default_cache_directory = "./cache"

input_params = {
    'dataset_name': 'NF'
}
splitter_params = {
    'num_samples_per_class': 20,
    'test_size': 1000,
    'seed': 345084,
}
node2vec_params = {
    "p": 1.0, "q": 1.0,
    "dimensions": 256,
    'num_walks': 2,  # Number of walks from each node
    'walk_length': 10,  # Walk length
    'window_size': 5,  # Context size for word2vec
    'iter': 1,  # number of SGD iterations (epochs)
    'workers': 4,  # number of workers for word2vec
    'weighted': False,  # are edges weighted?
    'directed': False,  # are edges directed?
}
inference_params = {
    # type of data to use for inference: 'with_attributes', 'with_embeddings', or 'with_metric',
    'method': 'logistic',  # inference method to use: one of logistic, rforest, kNN
    'k': 3,  # the number of nearest neighbors for kNN classifier
    'n': 10,  # number of decision trees for Random Forest (rforest) algorithm
    'gamma': 1.,  # 1/C parameter for logistic regression (controls the degree of regularization)
    'penalty': 'l2',  # regularization type for logistic regression, one of 'l2' or 'l1'}
}

# ------------------------------------------------------------------------
# Task classes
# ------------------------------------------------------------------------

NFConverter = se.local_task_class(preprocess_nf,
                                  ['in_edgelist'])
Node2Vec = se.local_task_class(Node2Vec_task,
                                  ['in_vector'])

# EPGMConverter = se.http_task_class('http://localhost:5000/',
#                                    'epgm_converter',
#                                    ['in_epgm'])
#
# EPGMWriter = se.http_task_class('http://localhost:5000/',
#                                 'epgm_writer',
#                                 ['in_graph_conv', 'in_pred', 'in_epgm'])
#
# NodeSplitter = se.http_task_class('http://localhost:5000/',
#                                   'node_splitter',
#                                   ['in_graph_conv'])
#
# Node2Vec = se.http_task_class('http://localhost:5000/',
#                             'representation_learning',
#                             ['in_vector'])
#
# Inference = se.http_task_class('http://localhost:5000/',
#                                'inference',
#                                ['in_data', 'in_features'])

# ------------------------------------------------------------------------
# Workflow class
# ------------------------------------------------------------------------

class Node2VecWorkflow(se.Workflow):

    merge_inputs = True

    def workflow(self):
        input_graph = self.add_task(se.LocalDataset,
                                    location="nf_edges_1-1000.csv",
                                    params=input_params)

        nf_converter = self.add_task(NFConverter)
        nf_converter.in_edgelist = input_graph.out_json

        node2vec = self.add_task(Node2Vec, params=node2vec_params)
        node2vec.in_vector = nf_converter.out_json

        return node2vec

# ------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Stellar workflow for the example")
    parser.add_argument('-o', '--output', nargs='?', type=str, default=default_cache_directory,
                        help="The cache dir to store artefacts")
    args, cmdline_args = parser.parse_known_args()

    # Ensure the cache dir exists
    se.set_base_directory(args.output)

    sl.run_local(main_task_cls=Node2VecWorkflow, cmdline_args=cmdline_args)
