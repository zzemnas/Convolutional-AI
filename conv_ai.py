import tensorflow as tf
import tensorflow_datasets as tfds
dmlab = tfds.load("dmlab", as_supervised=True, shuffle_files=True)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
  
def augment(image, label):
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_hue(image, 0.1)
    return image, label
 
dmlab_train = dmlab['train'].shuffle(1000).batch(32).map(preprocess).map(augment).prefetch(tf.data.experimental.AUTOTUNE)
# pipeline

from tensorflow.keras import layers, activations, applications
from tensorflow.keras.models import Sequential

model=Sequential(
    layers=[
        layers.Input((360,480,3)),

        tf.keras.layers.Conv2D(32, kernel_size=8, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

        tf.keras.layers.Conv2D(64, kernel_size=8, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

        tf.keras.layers.Conv2D(128, kernel_size=8, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),

        tf.keras.layers.Conv2D(256, kernel_size=8, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        layers.GlobalAveragePooling2D(),
        layers.Dense(4000), 
        # layers.Dropout(0.3),
        layers.Dropout(0.5),
        layers.Dense(6)
        ]
)

from tensorflow.keras import losses, optimizers, metrics, callbacks

odel.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=optimizers.Adam(),
              metrics=[metrics.SparseCategoricalAccuracy()])

model.fit(dmlab_train, epochs=5, callbacks=[tb_callback])

# integrated gradients interpretation

pip install tf-explain
import numpy
from tf_explain.core.integrated_gradients import IntegratedGradients
dmlab_test = dmlab['test'].map(preprocess)

image, label = next(iter(dmlab_test))

image=tf.reshape(image,(1,360,480,3))
tf.nn.softmax(model.predict(image)).numpy()

max_lab = numpy.argmax(tf.nn.softmax(model.predict(image)).numpy())
# optimal label according to softmax

max_lab
label.numpy()
explainer = IntegratedGradients()
grid_max = explainer.explain((image, None), model, class_index=max_lab, n_steps=100)
import matplotlib.pyplot as plt
plt.imshow(image[0,:,:,:])
# the image
plt.imshow(grid_max)


