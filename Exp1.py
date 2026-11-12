import numpy as np
from tensorflow.keras import Sequential, layers, optimizers, utils

utils.set_random_seed(42)
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = Sequential([
    layers.Input((2,)),
    layers.Dense(4, activation="tanh"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer=optimizers.Adam(0.03),
              loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x, y, epochs=300, verbose=0)

_, accuracy = model.evaluate(x, y, verbose=0)
predictions = model.predict(x, verbose=0).ravel().round().astype(int)
print("XOR predictions:", predictions.tolist())
print(f"Accuracy: {accuracy:.0%}")