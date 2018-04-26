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
from baseline_recommender_plugin import baseline_recommender, baseline_predictor, baseline_metrics
from node2vec_plugin import Node2Vec_task
from link_prediction_plugin import link_regression

# Use sciluigi logger
logger = logging.getLogger('sciluigi-interface')

# ------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------
default_cache_directory = "./cache"

baseline_recommender_params = {
    "dataset_name": "ml_100k",
    "columns": ["user", "item", "rating", "timestamp"],
    "sep": "\t",
    "algorithm": "SVD++"
}
baseline_predict_params = {
    "columns": ["user", "item", "rating", "timestamp"],
    "sep": "\t"
}
movielens_conv_params = {
    "dataset_name": "ml_100k",
    "columns": ["uId", "mId", "score"],
    "usecols": [0, 1, 2],
    "sep": "\t"
}
netflix_conv_params = {
    "dataset_name": "nf_1000",
    "columns": ["mId", "uId", "score"],
    "usecols": [0, 1, 3],
    "sep": ","
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
link_regression_params = {
    'algorithm': 'nn_concat',
    'nn_dim': [32],
    'nn_activation': 'sigmoid'
}

# ------------------------------------------------------------------------
# Task classes
# ------------------------------------------------------------------------

NFConverter = se.local_task_class(preprocess_nf,
                                  ['in_edgelist'])
Node2Vec = se.local_task_class(Node2Vec_task,
                                  ['in_vector'])
LinkRegression = se.local_task_class(link_regression,
                                  ['in_emb'])
BaselineRecommender = se.local_task_class(baseline_recommender,
                                  ['in_data'])
BaselinePredictor = se.local_task_class(baseline_predictor,
                                  ['in_model', 'in_test'])
BaselineMetrics = se.local_task_class(baseline_metrics, ['in_pred'])

# ------------------------------------------------------------------------
# Workflow class
# ------------------------------------------------------------------------

class Node2VecWorkflow(se.Workflow):

    merge_inputs = True

    def workflow(self):
        graph_filename = os.path.expanduser("~/Code/Data/ml-100k/u.data")

        outputs = []
        for dimensions in [50, 100, 150]:
            for walk_length in [10, 20]:
                for iter in [1, 4]:
                    for algorithm in ['nn_concat', 'nn_mul']:
                        for activation in ['relu', 'sigmoid', 'linear']:
                            wf = self.workflow_single(graph_filename,
                                                 dimensions=dimensions,
                                                 walk_length=walk_length,
                                                 iter=iter,
                                                 algorithm=algorithm,
                                                 activation=activation)
                            outputs.append(wf.out_json)

        performance = self.add_task(
            se.DisplayPerformance,
            display_params=['dimensions', 'walk_length', 'algorithm', 'iter', 'algorithm', 'activation'],
            display_metrics={'Test RMSE': 'test_error', 'Train RMSE': 'train_error'},
        )
        performance.in_json = outputs
        return performance

    def workflow_single(self, graph_filename, p=1, q=1, dimensions=256, walk_length=10, iter=1, algorithm='nn_concat', activation='sigmoid', nn_dim=64):
        node2vec_params['p'] = p
        node2vec_params['q'] = q
        node2vec_params['dimensions'] = dimensions
        node2vec_params['walk_length'] = walk_length
        node2vec_params['iter'] = iter

        link_regression_params['algorithm'] = algorithm
        link_regression_params['nn_activation'] = activation
        link_regression_params['nn_dim'] = [nn_dim]

        input_graph = self.add_task(se.LocalDataset,
                                    location=graph_filename)

        nf_converter = self.add_task(NFConverter,
                                     params=movielens_conv_params)
        nf_converter.in_edgelist = input_graph.out_json

        # Split edges


        # Node2vec embeddings
        node2vec = self.add_task(Node2Vec, params=node2vec_params)
        node2vec.in_vector = nf_converter.out_json

        # Train prediction
        prediction = self.add_task(LinkRegression, params=link_regression_params)
        prediction.in_emb = node2vec.out_json
        prediction.in_graph = nf_converter.out_json

        return prediction

class BaselineWorkflow(se.Workflow):

    merge_inputs = True

    def workflow(self):
        train_filename = os.path.expanduser("~/Code/Data/ml-100k/u1.base")
        train_data = self.add_task(se.LocalDataset,
                                    location=train_filename)

        test_filename = os.path.expanduser("~/Code/Data/ml-100k/u1.test")
        test_data = self.add_task(se.LocalDataset,
                                    location=test_filename)

        recommender = self.add_task(BaselineRecommender,
                                    params=baseline_recommender_params)
        recommender.in_data = train_data.out_json

        predict = self.add_task(BaselinePredictor,
                                    params=baseline_predict_params)
        predict.in_test = test_data.out_json
        predict.in_model = recommender.out_json

        metrics = self.add_task(BaselineMetrics)
        metrics.in_pred = predict.out_json

        output = self.add_task(
            se.DisplayPerformance,
            display_params=['dataset_name'],
            display_metrics={'RMSE':'rmse'},
        )
        output.in_json = metrics.out_json

        return output


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
