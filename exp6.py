import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, datasets, layers, utils

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

fig, (accuracy_plot, loss_plot) = plt.subplots(1, 2, figsize=(9, 3))
epochs = 10
for use_bn in (False, True):
    utils.set_random_seed(42)
    model = Sequential([layers.Input((28, 28)), layers.Flatten()])
    for units in (128, 64):
        model.add(layers.Dense(units, activation="relu"))
        if use_bn:
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=64,
                        validation_split=0.2, verbose=0)

    name = "Dropout + BatchNorm" if use_bn else "Dropout"
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name}: {accuracy:.2%} test accuracy")

    steps = range(1, epochs + 1)
    accuracy_plot.plot(steps, history.history["val_accuracy"], label=name)
    loss_plot.plot(steps, history.history["val_loss"], label=name)

accuracy_plot.set(title="Validation accuracy", xlabel="Epoch")
loss_plot.set(title="Validation loss", xlabel="Epoch")
accuracy_plot.legend()
loss_plot.legend()
fig.tight_layout()
plt.show()