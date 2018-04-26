import numpy as np

from sklearn import model_selection, linear_model, ensemble, svm, metrics

from tensorflow.contrib.keras import layers, models, optimizers, utils
from tensorflow.contrib.keras import backend as K

import logging
logger = logging.getLogger('sciluigi-interface')


def RMSE(y_true, y_pred):
    return K.sqrt(K.mean((y_true - y_pred)**2))


def nn_regression(n_features, layer_dim=[32], act='sigmoid', merge_method='mul'):
    inputs_a = layers.Input(shape=(n_features,))
    inputs_b = layers.Input(shape=(n_features,))

    xa = layers.Dense(layer_dim[0], activation=act)(inputs_a)
    xb = layers.Dense(layer_dim[0], activation=act)(inputs_b)

    if merge_method == 'mul':
        le = layers.Multiply()([xa,xb])
    elif merge_method == 'concat':
        le = layers.Concatenate()([xa, xb])

    predictions = layers.Dense(1, activation='linear')(le)

    optimizer = optimizers.SGD(lr=0.1, clipnorm=5)

    model = models.Model(inputs=[inputs_a, inputs_b], outputs=predictions)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=[RMSE])

    return model


def nn_classifier(n_features, n_output, merge_method='mul'):
    inputs_a = layers.Input(shape=(n_features,))
    inputs_b = layers.Input(shape=(n_features,))

    xa = layers.Dense(64, activation='relu')(inputs_a)
    xb = layers.Dense(64, activation='relu')(inputs_b)

    if merge_method == 'mul':
        le = layers.Multiply()([xa,xb])
    elif merge_method == 'concat':
        le = layers.Concatenate()([xa, xb])

    predictions = layers.Dense(n_output, activation='softmax')(le)

    optimizer = optimizers.SGD(lr=0.1, clipnorm=5)

    model = models.Model(inputs=[inputs_a, inputs_b], outputs=predictions)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['mean_squared_error'])

    return model


def link_regression_binarized(handler, inputs, output, parameters):
    input_spec = inputs['in_data']
    dataset_name = input_spec['dataset_name']

    # Load data
    embeddings = handler.load_pickle(input_spec['embeddings_filename'])
    G = handler.load_pickle(input_spec['nx_filename'])

    def edge_binarizer(a, b):
        return a*b

    # Generate the binarized link features
    N_edges = len(G.edges)
    N_emb = embeddings.shape[1]
    edge_features = np.zeros((N_edges, N_emb), dtype='float32')
    edge_labels = np.zeros(N_edges, dtype='float32')
    for ii, (u_node, m_node) in enumerate(G.edges):
        u_emb = embeddings[u_node]
        m_emb = embeddings[m_node]
        edge_features[ii] = edge_binarizer(u_emb, m_emb)
        edge_labels[ii] = G.edges[(u_node, m_node)]['score']

    logger.info("Training node embedding link regression using {}"
                .format(parameters['algorithm']))

    fit_args = {}
    if parameters['algorithm'] == 'linear':
        reg = linear_model.LinearRegression()
    elif parameters['algorithm'] == 'gbr':
        reg = ensemble.GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth = 1,
            random_state = 0,
            loss = 'ls'
        )
    elif parameters['algorithm'] == 'svr':
        reg = svm.SVR()

    # Split and train the recommender model
    splitter = model_selection.KFold(n_splits=5, shuffle=False)

    train_inx, test_inx = next(splitter.split(edge_features))
    train_features = edge_features[train_inx]
    train_labels = edge_labels[train_inx]

    reg.fit(train_features, train_labels, **fit_args)
    train_predictions = np.ravel(reg.predict(train_features))
    train_error = np.sqrt(np.mean((train_predictions - train_labels)**2))

    test_features = edge_features[test_inx]
    test_labels = edge_labels[test_inx]
    test_predictions = np.ravel(reg.predict(test_features))
    test_error = np.sqrt(np.mean((test_predictions - test_labels)**2))

    print("Train Error: {}  Test error: {}".format(train_error, test_error))

    # Save trained predictor
    #predictor_loc = handler.save_as_pickle(algo, "baseline_predictor.pkl")

    return {
        #"predictor": predictor_loc,
        "dataset_name": dataset_name
    }



def link_regression(handler, inputs, output, parameters):
    input_spec = inputs['in_data']
    dataset_name = input_spec['dataset_name']

    # Load data
    embeddings = handler.load_pickle(input_spec['embeddings_filename'])
    G = handler.load_pickle(input_spec['nx_filename'])

    # Generate the binarized link features
    N_edges = len(G.edges)
    N_emb = embeddings.shape[1]

    logger.info("Training node embedding link regression using {}"
                .format(parameters['algorithm']))

    nn_dim = parameters.get('nn_dim', [32])
    nn_act = parameters.get('nn_activation', 'sigmoid')

    fit_args = {
        'epochs': 20,
        'batch_size': 64,
        'validation_split': 0.0,
        'verbose': 2
    }
    if parameters['algorithm'] == 'linear':
        reg = linear_model.LinearRegression()
        fit_args = {}

    elif parameters['algorithm'] == 'nn_mul':
        reg = nn_regression(N_emb, nn_dim, nn_act, merge_method='mul')

    elif parameters['algorithm'] == 'nn_concat':
        reg = nn_regression(N_emb, nn_dim, nn_act, merge_method='concat')

    # Generate the edge features
    N_edges = len(G.edges)
    N_emb = embeddings.shape[1]
    edge_features_a = np.zeros((N_edges, N_emb), dtype='float32')
    edge_features_b = np.zeros((N_edges, N_emb), dtype='float32')
    edge_labels = np.zeros(N_edges, dtype='float32')
    split_label = np.zeros(N_edges, dtype='int32')
    for ii, edge in enumerate(G.edges):
        # Movie IDs are coded less than user Ids
        u_node, m_node = max(edge), min(edge)
        attr = G.edges[(u_node, m_node)]
        edge_features_a[ii] = embeddings[u_node]
        edge_features_b[ii] = embeddings[m_node]
        edge_labels[ii] = attr['score']
        split_label[ii] = attr['split']

    # Training split
    train_features_a = edge_features_a[split_label == 0]
    train_features_b = edge_features_b[split_label == 0]
    train_labels = edge_labels[split_label == 0]

    logger.info("Training on {} edges and {} embedding features"
                .format(train_labels.shape[0], N_emb))

    # Train the recommender model
    reg.fit([train_features_a, train_features_b], train_labels, **fit_args)
    train_predictions = np.ravel(
        reg.predict([train_features_a, train_features_b])
    )
    train_error = np.sqrt(np.mean((train_predictions - train_labels)**2))

    # Test split
    test_features_a = edge_features_a[split_label == 1]
    test_features_b = edge_features_b[split_label == 1]
    test_labels = edge_labels[split_label == 1]

    test_predictions = np.ravel(
        reg.predict([test_features_a, test_features_b])
    )
    test_error = np.sqrt(np.mean((test_predictions - test_labels)**2))

    logger.info("Train Error: {}  Test error: {}".format(train_error, test_error))

    # Save trained predictor
    predictor_weights = reg.get_weights()
    predictor_loc = handler.save_as_pickle(predictor_weights, "model_weights.pkl")

    return {
        "predictor": predictor_loc,
        "dataset_name": dataset_name,
        "train_error": train_error,
        "test_error": test_error,
    }

