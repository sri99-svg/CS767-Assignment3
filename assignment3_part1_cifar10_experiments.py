"""
assignment3_part1_cifar10_experiments.py

Part 1 of MET CS 767 Assignment 3:
- Baseline CNN on CIFAR-10 (reduced dataset)
- Modified CNN with depthwise-separable blocks, label smoothing, and Adam+cosine lr

"""

import tensorflow as tf
import numpy as np
from typing import Tuple

#config


TRAIN_FRACTION = 0.30
BATCH_SIZE = 128
EPOCHS_BASELINE = 20
EPOCHS_MODIFIED = 30
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)



#data loading & Subsampling


def load_cifar10_subset(train_fraction: float = 0.3) -> Tuple[tuple, tuple]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    #normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    #one hot encode
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    #subsample training data
    n_train = int(train_fraction * x_train.shape[0])
    idx = np.random.permutation(x_train.shape[0])[:n_train]

    x_train_sub = x_train[idx]
    y_train_sub = y_train[idx]

    print(f"Using {n_train} training samples out of {x_train.shape[0]} total.")
    return (x_train_sub, y_train_sub), (x_test, y_test)



#baseline CNN


def build_baseline_cnn(input_shape=(32, 32, 3), num_classes: int = 10) -> tf.keras.Model:
    """
    Reasonably standard small CNN baseline.
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="baseline_cifar10_cnn")
    return model


def train_baseline(x_train, y_train, x_test, y_test):
    model = build_baseline_cnn()

    opt = tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.9, nesterov=True
    )

    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_BASELINE,
        validation_data=(x_test, y_test),
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[Baseline] Test accuracy: {test_acc:.4f}")
    return model, history, test_acc



# Modified CNN (depthwise + reg + Adam+cosine)

def ds_block(x, filters, kernel_size=3, strides=1):
    """
    Depthwise-separable conv block with BatchNorm + ReLU.
    """
    x = tf.keras.layers.SeparableConv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def build_modified_cnn(input_shape=(32, 32, 3), num_classes: int = 10) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)


    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = tf.keras.layers.RandomContrast(0.1)(x)


    x = ds_block(x, 32)
    x = ds_block(x, 32)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = ds_block(x, 64)
    x = ds_block(x, 64)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)


    x = tf.keras.layers.Conv2D(96, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(96, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="modified_cifar10_cnn")
    return model


def train_modified(x_train, y_train, x_test, y_test):
    model = build_modified_cnn()


    steps_per_epoch = len(x_train) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS_MODIFIED
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=total_steps
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_MODIFIED,
        validation_data=(x_test, y_test),
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[Modified] Test accuracy: {test_acc:.4f}")
    return model, history, test_acc



#main entry point

def main():
    (x_train, y_train), (x_test, y_test) = load_cifar10_subset(TRAIN_FRACTION)

    print("\n=== Training baseline CNN ===")
    _, _, baseline_acc = train_baseline(x_train, y_train, x_test, y_test)

    print("\n=== Training modified CNN ===")
    _, _, modified_acc = train_modified(x_train, y_train, x_test, y_test)

    print("\nComparison:")
    print(f"Baseline test accuracy: {baseline_acc:.4f}")
    print(f"Modified test accuracy: {modified_acc:.4f}")


if __name__ == "__main__":
    main()
