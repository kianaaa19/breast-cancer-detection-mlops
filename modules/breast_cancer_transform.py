"""
breast_cancer_transform.py
"""
import tensorflow as tf

LABEL_KEY = "diagnosis"
FEATURE_KEYS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean"
]

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features.

    Args:
        inputs: map from feature keys to raw features.

    Return:
        outputs: map from feature keys to transformed features.
    """
    outputs = {}

    # Normalize numeric features
    for key in FEATURE_KEYS:
        outputs[transformed_name(key)] = tf.math.divide_no_nan(
            tf.cast(inputs[key], tf.float32),
            tf.reduce_max(tf.cast(inputs[key], tf.float32))
        )

    # Convert 'diagnosis' labels (M/B) to binary format (1/0)
    outputs[transformed_name(LABEL_KEY)] = tf.where(
        tf.equal(inputs[LABEL_KEY], "M"), 1, 0
    )

    return outputs
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
