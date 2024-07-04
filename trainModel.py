import tensorflow_datasets as tfds
import tensorflow as tf
import os


class Model:
    def __init__(self):
        if os.path.exists("./model.keras"):
            self.model = tf.keras.models.load_model("./model.keras")
        else:
            # Define Model
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10)
            ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def normaliseImg(self, image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    def trainModel(self, epochs, dsTrain, dsTest, dsInfo):
        # Train
        dsTrain = dsTrain.map(
            self.normaliseImg, num_parallel_calls=tf.data.AUTOTUNE
        )
        dsTrain = dsTrain.cache()
        dsTrain = dsTrain.shuffle(dsInfo.splits["train"].num_examples)
        dsTrain = dsTrain.batch(128)
        dsTrain = dsTrain.prefetch(tf.data.AUTOTUNE)

        # Test
        dsTest = dsTest.map(
            self.normaliseImg, num_parallel_calls=tf.data.AUTOTUNE
        )
        dsTest = dsTest.batch(128)
        dsTest = dsTest.cache()
        dsTest = dsTest.prefetch(tf.data.AUTOTUNE)

        with tf.device('/CPU:0'):
            self.model.fit(
                dsTrain,
                epochs=epochs,
                validation_data=dsTest
            )

        self.model.save("model.keras")

    def predict(self, image):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, 0)
        predictions = self.model(image)
        return tf.argmax(predictions, axis=1).numpy()


if __name__ == "__main__":
    # get the training data
    (dsTrain, dsTest), dsInfo = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    trainingModel = Model()
    epochs = int(input("How many epochs to train: "))
    trainingModel.trainModel(epochs, dsTrain, dsTest, dsInfo)
