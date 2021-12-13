from animal_recognition import load_data, batch_size
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 classes
categories = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

# загрузим тестовый набор
ds_train, ds_test, info = load_data()
# загрузим итоговую модель с весовыми коэффициентами
model = load_model("results/cifar10-model-v1.h5")


# оценка
loss, accuracy = model.evaluate(x_test, y_test)
print("Тестовая оценка:", accuracy*100, "%")


# получить прогноз для случайного изображения
data_sample = next(iter(ds_test))
sample_image = data_sample[0].numpy()[0]
sample_label = categories[data_sample[1].numpy()[0]]
prediction = np.argmax(model.predict(sample_image.reshape(-1, *sample_image.shape))[0])
print("Predicted label:", categories[prediction])
print("True label:", sample_label)