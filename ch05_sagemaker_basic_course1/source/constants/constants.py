DEPENDENCIES = ["catboost"]  # This is not used at the endpoint.
# It is used while launching the job to decide which all dependency packages
# are to be uploaded to the training container.

INPUT_DATA_FILENAME = "data.csv"
TRAIN_CHANNEL = "train"
VALIDATION_CHANNEL = "validation"

TRAIN_VALIDATION_FRACTION = 0.2
RANDOM_STATE_SAMPLING = 200

# Optimizer constants
DEFAULT_ITERATIONS = 500
DEFAULT_EARLY_STOPPING_ROUNDS = 5
DEFAULT_EVAL_METRIC = "Auto"
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_DEPTH = 6
DEFAULT_L2_LEAF_REG = 3
DEFAULT_RANDOM_STRENGTH = 32
DEFAULT_MAX_LEAVES = 31
DEFAULT_RSM = 1
DEFAULT_SAMPLING_FREQUENCY = "PerTreeLevel"
DEFAULT_MIN_DATA_IN_LEAF = 1
DEFAULT_BAGGING_TEMPERATURE = 1
DEFAULT_BOOSTING_TYPE = "Auto"
DEFAULT_SCALE_POS_WEIGHT = 1
DEFAULT_MAX_BIN = "Auto"
DEFAULT_RANDOM_SEED = 0
DEFAULT_THREAD_COUNT = -1
DEFAULT_VERBOSE = 1
DEFAULT_GROW_POLICY = "SymmetricTree"

# Problem type - objective mapping
CLASSIFICATION_PROBLEM_TYPE_OBJECTIVE_MAPPING = {
    "binary classification": "Logloss",
    "multi-class classification": "MultiClass",
}

# Problem type - evaluation metric mapping
CLASSIFICATION_PROBLEM_TYPE_METRIC_MAPPING = {
    "binary classification": "AUC",
    "multi-class classification": "MultiClass",
}

IS_TRAINED_ON_INPUT_DATA = "is_trained_on_input_data"
INPUT_MODEL_UNTARRED_PATH = "_input_model_extracted/"

MODEL_INFO_FILE_NAME = "__models_info__.json"

VALIDATION_SCORE = "validation"
