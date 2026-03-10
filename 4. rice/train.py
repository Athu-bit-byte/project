import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------
# DATASET PATH
# ---------------------------------------------------------
DATASET_PATH = r"C:\Users\allen\Downloads\rice\train"
MODEL_SAVE_PATH = r"C:\Users\allen\Downloads\rice\model_rice_disease.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 4


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
def load_dataset(path):
    filepaths = []
    labels = []

    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)

        if os.path.isdir(class_path):
            for img in os.listdir(class_path):
                if img.lower().endswith((".jpg", ".png", ".jpeg")):
                    filepaths.append(os.path.join(class_path, img))
                    labels.append(class_name)

    return pd.DataFrame({"Filepath": filepaths, "Label": labels})


# ---------------------------------------------------------
# Hybrid Model
# ---------------------------------------------------------
def build_hybrid_model(num_classes):

    effnet = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    effnet.trainable = False

    convnext = tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    convnext.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x1 = effnet(inputs)
    x2 = convnext(inputs)

    x = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------
def train():

    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)

    train_df, val_df = train_test_split(
        df,
        train_size=0.8,
        stratify=df["Label"],
        random_state=42
    )

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        horizontal_flip=True
    )

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    train_images = train_gen.flow_from_dataframe(
        train_df,
        x_col="Filepath",
        y_col="Label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_images = test_gen.flow_from_dataframe(
        val_df,
        x_col="Filepath",
        y_col="Label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    print("Classes:", train_images.class_indices)

    model = build_hybrid_model(len(train_images.class_indices))

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Training...")
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=EPOCHS
    )

    print("Saving model...")
    model.save(MODEL_SAVE_PATH)

    # Accuracy plot
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.legend(["Train", "Val"])
    plt.title("Accuracy")
    plt.savefig(r"C:\Users\allen\Downloads\rice\accuracy.png")
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["Train", "Val"])
    plt.title("Loss")
    plt.savefig(r"C:\Users\allen\Downloads\rice\loss.png")
    plt.close()


train()
