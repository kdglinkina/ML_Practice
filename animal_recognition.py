from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow_datasets as tfds
import os

# Гиперпараметры
batch_size = 64
# 10 категорий для изображений  (CIFAR-10)
num_classes = 10
# количество эпох для обучения
epochs = 30

def load_data():
    # загружаем набор данных CIFAR-10 dataset и делает предварительную обработку

    def preprocess_image(image, label):
        # преобразуем целочисленный диапазон [0, 255] в диапазон действительных чисел [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label
    # загружаем набор данных CIFAR-10, разделяем его на обучающий и тестовый
    ds_train, info = tfds.load("cifar10", with_info=True, split="train", as_supervised=True)
    ds_test = tfds.load("cifar10", split="test", as_supervised=True)
    # повторять набор данных, перемешивая, предварительно обрабатывая, разделяем по пакетам
    ds_train = ds_train.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    ds_test = ds_test.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    return ds_train, ds_test, info


# построение модели
def create_model(input_shape):
    # построение модели
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # сглаживание неровностей
    model.add(Flatten())
    # полносвязный слой
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    # печатаем итоговую архитектуру модели
    model.summary()
    # обучение модели с помощью оптимизатора Адама
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# обучение модели
if __name__ == "__main__":
    # загружаем данные
    ds_train, ds_test, info = load_data()
    # конструируем модель
    model = create_model(input_shape=info.features["image"].shape)
    # несколько хороших обратных вызовов
    logdir = os.path.join("logs", "cifar10-model-v1")
    tensorboard = TensorBoard(log_dir=logdir)
    # убедимся, что папка с результатами существует
    if not os.path.isdir("results"):
        os.mkdir("results")
    # обучаем
    model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=1,
              steps_per_epoch=info.splits["train"].num_examples // batch_size,
              validation_steps=info.splits["test"].num_examples // batch_size,
              callbacks=[tensorboard])
    # сохраняем модель на диске
    model.save("results/cifar10-model-v1.h5")