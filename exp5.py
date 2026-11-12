import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import img_to_array, load_img

vgg = VGG16(weights="imagenet", include_top=False)
model = Model(vgg.input, vgg.get_layer("block1_conv1").output)

image_path = get_sample_data("grace_hopper.jpg", asfileobj=False)
image = load_img(image_path, target_size=(224, 224))
x = preprocess_input(img_to_array(image)[None, ...])
maps = model.predict(x, verbose=0)[0]

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(maps[:, :, i], cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()