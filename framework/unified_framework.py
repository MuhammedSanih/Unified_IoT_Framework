from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model

from framework.aggregation import aggregate_prediction
from framework.data_router import identify_data_type
from framework.logger import log_prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CNN_MODEL_PATH = PROJECT_ROOT / "models" / "cifar10_model.keras"
RNN_MODEL_PATH = PROJECT_ROOT / "models" / "unsw_model.keras"

IMG_SIZE = (32, 32)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

NETWORK_CLASSES = [
    "Normal",
    "Attack"
]

print("\nLoading models...")

cnn_model = load_model(CNN_MODEL_PATH)
rnn_model = load_model(RNN_MODEL_PATH)

print("\nModels loaded successfully.")


def predict(input_data):

    data_type = identify_data_type(input_data)

    if data_type == "unknown":
        raise ValueError("Unsupported input type.")

    start = time.perf_counter()

    if data_type == "image":

        image = tf.keras.utils.load_img(
            input_data,
            target_size=IMG_SIZE
        )

        image = tf.keras.utils.img_to_array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        model_name = "CNN"

        probabilities = cnn_model.predict(image, verbose=0)

        index = np.argmax(probabilities)

        prediction = CLASS_NAMES[index]

        confidence = float(probabilities[0][index])

    else:

        sample = np.asarray(input_data, dtype=np.float32)

	# Convert (194,) -> (1,194)
        if sample.ndim == 1:
                sample = np.expand_dims(sample, axis=0)

	# Convert (1,194) -> (1,1,194)
        sample = np.expand_dims(sample, axis=1)

        model_name = "RNN"

        probabilities = rnn_model.predict(sample, verbose=0)

        confidence = float(probabilities[0][0])

        if confidence >= 0.5:
                prediction = "Attack"
        else:
                prediction = "Normal"

        confidence = max(confidence, 1 - confidence)

    end = time.perf_counter()

    aggregated_result = aggregate_prediction(

    input_type=data_type,

    selected_model=model_name,

    prediction=prediction,

    confidence=confidence,

    inference_time_ms=(end-start)*1000

    )

    # Save prediction to CSV
    log_prediction(aggregated_result)

    # Return result to caller
    return aggregated_result
