import argparse
import json
import logging
import os
import pathlib
import tarfile
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from catboost import Pool
from constants import constants
from sagemaker_jumpstart_prepack_script_utilities.prepack_inference import copy_inference_code
from sagemaker_jumpstart_script_utilities import data_prep
from sagemaker_jumpstart_script_utilities import model_info


logging.basicConfig(level=logging.INFO)


def import_cat_index_list(cat_index_path: List[str], feature_dim: int) -> List[int]:
    """Read the categorical column index from a JSON file, return the list of categorical column index.

    The JSON file should be formatted such that the key is 'cat_index' and value is a list of categorical column index.

    Args:
        cat_index_path: a list of path string that indicates the directory saving the categorical column index info.
        feature_dim (int): dimension of predicting features.

    Returns:
        list of categorical columns indexes.
    """
    assert (
        len(cat_index_path) == 1
    ), "Found json files for categorical indexes more than 1. Please ensure there is only one json file."

    with open(cat_index_path[0], "r") as f:
        cat_index_dict = json.loads(f.read())

    if cat_index_dict is not None:
        cat_index_list = list(cat_index_dict.values())[0]  # convert dict.values() from an iterable to a python list.
        if not cat_index_list:  # abort early if it is an empty list with no indexes for categorical columns
            return []

        assert all(isinstance(index, int) for index in cat_index_list), (
            f"Found non-integer index for categorical feature. "
            f"Please ensure each index is an integer from 1 to {feature_dim}."
        )
        assert all(1 <= index <= feature_dim for index in cat_index_list), (
            f"Found index for categorical feature smaller than 1 or more than the index of "
            f"last feature {feature_dim}."
        )
        return [index - 1 for index in cat_index_list]  # offset by 1 as the first column in the input is the target.
    else:
        return []


def prepare_data(
    train_dir: str, validation_dir: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read data from train and validation channel, and return predicting features and target variables.

    Args:
        train_dir (str): directory which saves the training data.
        validation_dir (str): directory which saves the validation data.

    Returns:
        Tuple of training features, training target, validation features, validation target.
    """

    if validation_dir is not None:
        logging.info(
            "Data in the validation channel is found. "
            "Reading the train and validation data from the training and validation channel, respectively."
        )
        df_train = pd.read_csv(data_prep.find_file_path(train_dir), header=None)
        df_train.columns = ["target"] + [f"feature_{x}" for x in range(df_train.shape[1] - 1)]

        df_validation = pd.read_csv(data_prep.find_file_path(os.path.join(validation_dir)), header=None)
        df_validation.columns = ["target"] + [f"feature_{x}" for x in range(df_validation.shape[1] - 1)]
    else:
        if os.path.exists(os.path.join(train_dir, constants.TRAIN_CHANNEL)):
            df_train = pd.read_csv(
                os.path.join(train_dir, constants.TRAIN_CHANNEL, constants.INPUT_DATA_FILENAME),
                header=None,
            )
            df_train = df_train.iloc[np.random.permutation(len(df_train))]
            df_train.columns = ["target"] + [f"feature_{x}" for x in range(df_train.shape[1] - 1)]

            try:
                df_validation = pd.read_csv(
                    os.path.join(train_dir, constants.VALIDATION_CHANNEL, constants.INPUT_DATA_FILENAME),
                    header=None,
                )
                df_validation.columns = ["target"] + [f"feature_{x}" for x in range(df_validation.shape[1] - 1)]

            except FileNotFoundError:  # when validation data is not available in the directory
                logging.info(
                    f"Validation data is not found. {constants.TRAIN_VALIDATION_FRACTION*100}% of training data is "
                    f"randomly selected as validation data. The seed for random sampling is "
                    f"{constants.RANDOM_STATE_SAMPLING}."
                )
                df_validation = df_train.sample(
                    frac=constants.TRAIN_VALIDATION_FRACTION,
                    random_state=constants.RANDOM_STATE_SAMPLING,
                )
                df_train.drop(df_validation.index, inplace=True)
                df_validation.reset_index(drop=True, inplace=True)
                df_train.reset_index(drop=True, inplace=True)
        else:
            df_train = pd.read_csv(data_prep.find_file_path(train_dir), header=None)
            df_train = df_train.iloc[np.random.permutation(len(df_train))]
            df_train.columns = ["target"] + [f"feature_{x}" for x in range(df_train.shape[1] - 1)]

            logging.info(
                f"Validation data is not found. {constants.TRAIN_VALIDATION_FRACTION * 100}% of training data is "
                f"randomly selected as validation data. The seed for random sampling is "
                f"{constants.RANDOM_STATE_SAMPLING}."
            )
            df_validation = df_train.sample(
                frac=constants.TRAIN_VALIDATION_FRACTION,
                random_state=constants.RANDOM_STATE_SAMPLING,
            )
            df_train.drop(df_validation.index, inplace=True)
            df_validation.reset_index(drop=True, inplace=True)
            df_train.reset_index(drop=True, inplace=True)

    X_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, :1]
    X_val, y_val = df_validation.iloc[:, 1:], df_validation.iloc[:, :1]

    return X_train, y_train, X_val, y_val


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    parser.add_argument("--pretrained-model", type=str, default=os.environ.get("SM_CHANNEL_MODEL"))
    parser.add_argument("--iterations", type=int, default=constants.DEFAULT_ITERATIONS)
    parser.add_argument("--early_stopping_rounds", type=int, default=constants.DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--eval_metric", type=str, default=constants.DEFAULT_EVAL_METRIC)
    parser.add_argument("--learning_rate", type=float, default=constants.DEFAULT_LEARNING_RATE)
    parser.add_argument("--depth", type=float, default=constants.DEFAULT_DEPTH)
    parser.add_argument("--l2_leaf_reg", type=float, default=constants.DEFAULT_L2_LEAF_REG)
    parser.add_argument("--random_strength", type=float, default=constants.DEFAULT_RANDOM_STRENGTH)
    parser.add_argument("--max_leaves", type=int, default=constants.DEFAULT_MAX_LEAVES)
    parser.add_argument("--rsm", type=float, default=constants.DEFAULT_RSM)
    parser.add_argument("--sampling_frequency", type=str, default=constants.DEFAULT_SAMPLING_FREQUENCY)
    parser.add_argument("--min_data_in_leaf", type=int, default=constants.DEFAULT_MIN_DATA_IN_LEAF)
    parser.add_argument("--bagging_temperature", type=float, default=constants.DEFAULT_BAGGING_TEMPERATURE)
    parser.add_argument("--boosting_type", type=str, default=constants.DEFAULT_BOOSTING_TYPE)
    parser.add_argument("--scale_pos_weight", type=float, default=constants.DEFAULT_SCALE_POS_WEIGHT)
    parser.add_argument("--max_bin", type=str, default=constants.DEFAULT_MAX_BIN)
    parser.add_argument("--grow_policy", type=str, default=constants.DEFAULT_GROW_POLICY)
    parser.add_argument("--random_seed", type=int, default=constants.DEFAULT_RANDOM_SEED)
    parser.add_argument("--thread_count", type=int, default=constants.DEFAULT_THREAD_COUNT)
    parser.add_argument("--verbose", type=int, default=constants.DEFAULT_VERBOSE)

    return parser.parse_known_args()


def save_model_info(input_model_untarred_path: str, model_dir: str) -> None:
    """Save model info to the output directory along with is_trained_on_input_data parameter set to True.

    Read the existing model_info file in input_model directory if exists, set is_trained_on_input_data parameter to
    True and saves it in the output model directory.
    Args:
        input_model_untarred_path: Input model is untarred into this directory.
        model_dir: Output model directory.
    """

    input_model_info_file_path = os.path.join(input_model_untarred_path, constants.MODEL_INFO_FILE_NAME)
    try:
        with open(input_model_info_file_path, "r") as f:
            model_info: Dict[str, Any] = json.load(f)
    except FileNotFoundError:
        logging.info(f"Info file not found at '{input_model_info_file_path}'.")
        model_info: Dict[str, Any] = {}
    except Exception as e:
        logging.error(f"Error parsing model_info file: {e}.")
        raise

    model_info[constants.IS_TRAINED_ON_INPUT_DATA] = True

    output_model_info_file_path = os.path.join(model_dir, constants.MODEL_INFO_FILE_NAME)
    with open(output_model_info_file_path, "w") as f:
        f.write(json.dumps(model_info))


def run_with_args(args):
    """Run training."""
    X_train, y_train, X_val, y_val = prepare_data(train_dir=os.path.join(args.train), validation_dir=args.validation)

    # get problem type (binary classification or multi-class classification) from y_train
    if len(np.unique(y_train.values)) == 2:
        problem_type = "binary classification"
    else:
        problem_type = "multi-class classification"

    if args.thread_count == 0:
        error = ValueError(
            "Hyperparameter thread_count is found to be 0. Please ensure thread_count "
            "is more than 0 or equal to -1. Value -1 means that the number of threads is equal"
            " to the number of processor cores."
        )
        logging.error(f"{error}")
        raise error

    if args.eval_metric == "Auto":
        eval_metric = constants.CLASSIFICATION_PROBLEM_TYPE_METRIC_MAPPING[problem_type]
    else:
        eval_metric = args.eval_metric

    # specify your configurations as a dict
    params = {
        "task_type": "CPU",  # TODO: support GPU training.
        "loss_function": constants.CLASSIFICATION_PROBLEM_TYPE_OBJECTIVE_MAPPING[problem_type],
        "iterations": args.iterations,
        "early_stopping_rounds": args.early_stopping_rounds,
        "eval_metric": eval_metric,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_strength": args.random_strength,
        "max_leaves": args.max_leaves,
        "rsm": args.rsm,
        "sampling_frequency": args.sampling_frequency,
        "min_data_in_leaf": args.min_data_in_leaf,
        "bagging_temperature": args.bagging_temperature,
        "boosting_type": None if args.boosting_type == "Auto" else args.boosting_type,
        "scale_pos_weight": args.scale_pos_weight if problem_type == "binary classification" else None,
        "max_bin": None if args.max_bin == "Auto" else int(args.max_bin),
        "grow_policy": args.grow_policy,
        "random_seed": args.random_seed,
        "thread_count": args.thread_count,
        "verbose": args.verbose,
    }

    # get categorical indexes
    cat_index_path = list(pathlib.Path(args.train).glob("*.json"))
    cat_features = None
    if cat_index_path:
        cat_index_list = import_cat_index_list(cat_index_path=cat_index_path, feature_dim=X_train.shape[1])
        if cat_index_list:
            params.update(cat_features=cat_index_list)
            cat_features = cat_index_list

    # create dataset for catboost
    cat_train, cat_eval = Pool(data=X_train, label=y_train, cat_features=cat_features), Pool(
        data=X_val, label=y_val, cat_features=cat_features
    )

    should_save_model = args.current_host == args.hosts[0]

    # train
    model = CatBoostClassifier(**params)
    input_model_path = next(pathlib.Path(args.pretrained_model).glob("*.tar.gz"))
    if not os.path.exists(constants.INPUT_MODEL_UNTARRED_PATH):
        os.mkdir(constants.INPUT_MODEL_UNTARRED_PATH)
    with tarfile.open(input_model_path, "r") as saved_model_tar:
        saved_model_tar.extractall(constants.INPUT_MODEL_UNTARRED_PATH)
    is_partially_trained = model_info.is_input_model_partially_trained(
        input_model_untarred_path=constants.INPUT_MODEL_UNTARRED_PATH
    )
    input_model: Optional[CatBoostClassifier] = None
    if is_partially_trained:
        logging.info("Using previously trained model as the starting model.")
        file_path = os.path.join(constants.INPUT_MODEL_UNTARRED_PATH, "model")
        input_model = CatBoostClassifier()
        input_model.load_model(file_path)
    model.fit(cat_train, eval_set=cat_eval, init_model=input_model)

    score = model.get_best_score()[constants.VALIDATION_SCORE][eval_metric]
    logging.info(f"{eval_metric}: {score}")

    if should_save_model:
        logging.info("Saving model...")
        # save model to file
        export_path = os.path.join(args.model_dir, "model")
        model.save_model(export_path)
        save_model_info(input_model_untarred_path=constants.INPUT_MODEL_UNTARRED_PATH, model_dir=args.model_dir)


if __name__ == "__main__":
    args, unknown = _parse_args()
    run_with_args(args)
    copy_inference_code(dst_path=args.model_dir)
