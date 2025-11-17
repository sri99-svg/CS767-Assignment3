"""
assignment3_part2_grocery_bill_cnn.py

Part 2 of MET CS 767 Assignment 3:
"Grocery Bill Line-Item Detector" CNN.

Expected directory structure (example):

data/
  train/
    milk/
    bread/
    spinach/
    ...
  val/
    milk/
    bread/
    spinach/
    ...

Each subfolder name under 'train' and 'val' is a class label.
"""

import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#config


DATA_DIR = "data"    
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# Data Pipeline

def create_generators(
    data_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32
):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=(0.8, 1.2),
        zoom_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    return train_gen, val_gen


#model architecture


def ds_block(x, filters, kernel_size=3, strides=1):

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


def build_grocery_cnn(
    input_shape=(224, 224, 3),
    num_classes: int = 4 
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)


    x = tf.keras.layers.RandomRotation(0.08)(inputs)
    x = tf.keras.layers.RandomZoom(0.1)(x)
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

    x = tf.keras.layers.Dense(128, activation="gelu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="grocery_bill_cnn")
    return model


#training


def train_grocery_model():
    train_gen, val_gen = create_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)

    num_classes = train_gen.num_classes
    print(f"Detected {num_classes} classes: {train_gen.class_indices}")

    model = build_grocery_cnn(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=num_classes
    )


    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    steps_per_epoch = train_gen.samples // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=total_steps
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        verbose=2
    )

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\n[Groceries] Validation accuracy: {val_acc:.4f}")

    return model, history, train_gen, val_gen


#inference helper


def predict_single_image(model, img_path: str, class_indices: dict):
    """
    Loads one image from disk and returns predicted class label
    """
    from tensorflow.keras.preprocessing import image

    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    class_id = np.argmax(preds, axis=1)[0]


    idx_to_class = {v: k for k, v in class_indices.items()}
    label = idx_to_class[class_id]
    confidence = float(np.max(preds))

    print(f"Prediction for {os.path.basename(img_path)}: {label} ({confidence:.3f})")
    return label, confidence


#main entry point


def main():
    model, history, train_gen, val_gen = train_grocery_model()


if __name__ == "__main__":
    main()
