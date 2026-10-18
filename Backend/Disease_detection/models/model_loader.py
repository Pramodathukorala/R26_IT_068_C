import os
import tensorflow as tf


# =========================================================
# MODEL DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        __file__
    )
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# =========================================================
# MODEL PATHS
# =========================================================

BODY_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "body_model.keras"
)

FECAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fecal_model.keras"
)

INPUT_VALIDATOR_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "input_validator.keras"
)


# =========================================================
# LOAD MODELS
# =========================================================

print("Loading AI models...")

body_model = tf.keras.models.load_model(
    BODY_MODEL_PATH
)

fecal_model = tf.keras.models.load_model(
    FECAL_MODEL_PATH
)

input_validator_model = tf.keras.models.load_model(
    INPUT_VALIDATOR_MODEL_PATH
)

print("All AI models loaded successfully!")


# =========================================================
# MODEL GETTERS
# =========================================================

def get_body_model():
    return body_model


def get_fecal_model():
    return fecal_model


def get_input_validator_model():
    return input_validator_model