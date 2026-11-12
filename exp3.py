import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential, layers

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
passengers = pd.read_csv(url)["Passengers"].to_numpy(dtype="float32")
window = 10
split = int(0.8 * len(passengers))
scale = passengers[:split].max()
values = passengers / scale

# Ten inputs and one target for each window
x = np.array([values[i - window:i] for i in range(window, len(values))])
x = x.reshape(-1, window, 1)
y = values[window:]
cut = split - window

model = Sequential([
    layers.Input((window, 1)),
    layers.Conv1D(32, 3, activation="relu"),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")
model.fit(x[:cut], y[:cut], epochs=30, shuffle=False, verbose=0)

actual = passengers[split:]
predicted = model.predict(x[cut:], verbose=0).ravel() * scale
print("Actual:", actual[:5].astype(int))
print("Predicted:", predicted[:5].round().astype(int))
print(f"Mean absolute error: {np.abs(actual - predicted).mean():.1f} passengers")

plt.plot(actual, label="Actual")
plt.plot(predicted, label="Predicted")
plt.xlabel("Test month")
plt.ylabel("Passengers")
plt.legend()
plt.show()