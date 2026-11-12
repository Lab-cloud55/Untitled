import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, datasets, layers

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255

model = Sequential([
    layers.Input((28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)

_, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.2%}")

predicted = model.predict(x_test[:1], verbose=0).argmax()
plt.imshow(x_test[0, :, :, 0], cmap="gray")
plt.title(f"Predicted {predicted} | Actual {y_test[0]}")
plt.axis("off")
plt.show()