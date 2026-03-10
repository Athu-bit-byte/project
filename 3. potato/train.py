import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------
# DATASET PATH
# ---------------------------------------------------------
TRAIN_PATH = r"C:\Users\allen\Downloads\potato\training"
VAL_PATH = r"C:\Users\allen\Downloads\potato\validation"
TEST_PATH = r"C:\Users\allen\Downloads\potato\testing"

MODEL_SAVE_PATH = r"C:\Users\allen\Downloads\potato\model_potato_disease.keras"

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

    print("Loading datasets...")

    train_df = load_dataset(TRAIN_PATH)
    val_df = load_dataset(VAL_PATH)
    test_df = load_dataset(TEST_PATH)

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

    test_images = test_gen.flow_from_dataframe(
        test_df,
        x_col="Filepath",
        y_col="Label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
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

    print("Evaluating...")
    loss, acc = model.evaluate(test_images, verbose=0)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    # Accuracy plot
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.legend(["Train", "Val"])
    plt.title("Accuracy")
    plt.savefig(r"C:\Users\allen\Downloads\potato\accuracy.png")
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["Train", "Val"])
    plt.title("Loss")
    plt.savefig(r"C:\Users\allen\Downloads\potato\loss.png")
    plt.close()


train()
