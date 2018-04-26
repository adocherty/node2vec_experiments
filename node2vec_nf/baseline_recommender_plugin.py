import numpy as np

from surprise import Dataset, Reader, SVD, SVDpp, KNNBasic, KNNWithMeans

import logging
logger = logging.getLogger('sciluigi-interface')

def baseline_recommender(handler, inputs, output, parameters):

    input_spec = inputs['in_data']
    train_file = input_spec['location']
    dataset_name = parameters['dataset_name']

    columns = parameters.get("columns", ['mId', 'uId', 'score'])
    sep = parameters.get("sep", ",")

    # Load dataset file
    line_format = " ".join(columns)
    reader = Reader(line_format=line_format, sep=sep, rating_scale=(1,5))
    data = Dataset.load_from_file(train_file, reader=reader)
    trainset = data.build_full_trainset()

    logger.info("Training baseline recommender using {}"
                .format(parameters['algorithm']))

    if parameters['algorithm'] == 'SVD':
        algo = SVD()
    elif parameters['algorithm'] == 'SVD++':
        algo = SVDpp()
    elif parameters['algorithm'] == 'KNN':
        algo = KNNBasic()
    elif parameters['algorithm'] == 'KNNmeans':
        algo = KNNWithMeans()

    # Train the recommender model
    algo.fit(trainset)

    # Save trained predictor
    predictor_loc = handler.save_as_pickle(algo, "baseline_predictor.pkl")

    return {
        "predictor": predictor_loc,
        "dataset_name": dataset_name
    }

def baseline_predictor(handler, inputs, output, parameters):
    input_spec = inputs['in_data']
    test_file = input_spec['location']
    dataset_name = input_spec['dataset_name']

    columns = parameters.get("columns", ['mId', 'uId', 'score'])
    sep = parameters.get("sep", ",")

    # Load test dataset file
    line_format = " ".join(columns)
    reader = Reader(line_format=line_format, sep=sep, rating_scale=(1,5))
    test_data = Dataset.load_from_file(test_file, reader=reader)
    testset = test_data.construct_testset(test_data.raw_ratings)

    # Predict ratings
    predictor = handler.load_pickle(input_spec['predictor'])
    predictions = predictor.test(testset, verbose=False)

    mse = np.mean([float((true_r - est)**2)
                   for (_, _, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)
    logger.info("[B] Test error: RMSE={}".format(rmse_))

    gt_ratings = np.array([x[2] for x in predictions])
    pred_ratings = np.array([x[3] for x in predictions])

    # Return predictions & ground truth
    predictions_loc = handler.save_as_pickle(pred_ratings, "test_predictions.pkl")
    ratings_loc = handler.save_as_pickle(gt_ratings, "test_labels.pkl")

    return {
        "predictions": predictions_loc,
        "true_ratings": ratings_loc,
        "dataset_name": dataset_name
    }


def baseline_metrics(handler, inputs, output, parameters):
    input_spec = inputs['in_data']
    dataset_name = input_spec['dataset_name']


    # Predicitons
    predictions = handler.load_pickle(input_spec['predictions'])
    labels = handler.load_pickle(input_spec['true_ratings'])

    # Metrics
    RMSE = np.sqrt(np.mean((predictions - labels)**2))

    return {
        "rmse": RMSE,
    }
