"""
tuner_module_file.py
"""
from typing import NamedTuple, Dict, Text, Any
import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from keras_tuner.engine import base_tuner
from tfx.components.trainer.fn_args_utils import FnArgs

# Constants
LABEL_KEY = "diagnosis"
FEATURE_KEYS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean"
]
NUM_EPOCHS = 10

TunerFnResult = NamedTuple("TunerFnResult", [
    ("tuner", base_tuner.BaseTuner),
    ("fit_kwargs", Dict[Text, Any]),
])

# Early stopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_binary_accuracy",
    mode="max",
    verbose=1,
    patience=10,
)

# Utility for transformed feature name
def transformed_name(key):
    return f"{key}_xf"

# Input function for loading dataset
def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=32):
    transform_feature_spec = tf_transform_output.transformed_feature_spec()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset

# Model builder for tuning
def model_builder(hp):
    inputs = {
        key: tf.keras.Input(shape=(1,), name=transformed_name(key), dtype=tf.float32)
        for key in FEATURE_KEYS
    }
    
    # Concatenate all feature inputs
    concatenated_inputs = tf.keras.layers.concatenate(list(inputs.values()))
    
    # Hyperparameter options
    num_hidden_layers = hp.Choice("num_hidden_layers", [1, 2, 3])
    dense_units = hp.Int("dense_units", min_value=16, max_value=128, step=16)
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
    learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    
    # Build dense layers
    x = concatenated_inputs
    for _ in range(num_hidden_layers):
        x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    # Compile model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    return model

# Tuner function
def tuner_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files[0], tf_transform_output, NUM_EPOCHS)
    eval_dataset = input_fn(fn_args.eval_files[0], tf_transform_output, NUM_EPOCHS)

    tuner = kt.Hyperband(
        hypermodel=model_builder,
        objective="val_binary_accuracy",
        max_epochs=NUM_EPOCHS,
        factor=3,
        directory=fn_args.working_dir,
        project_name="kt_hyperband",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [early_stopping_callback],
            "x": train_dataset,
            "validation_data": eval_dataset,
        },
    )
