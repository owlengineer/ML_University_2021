import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import losses
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.python.keras.backend as K

from sklearn.model_selection import train_test_split

from PIL import Image
import os, os.path

def loadImgsFromFolder(path):
    """Функция подгрузки в память массива картинок"""
    imgs = []
    valid_images = [".png"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img = np.asarray(Image.open(os.path.join(path, f)).resize((128, 128)))
        if img.ndim == 3:
            img = img[:, :, 1]
        imgs.append(img)
    return imgs


def mean_iou(y_true, y_pred):
    """Функция-метрика среднее пересечение для сегментации"""
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

class SaltDetectorProto(keras.Model):
    def __init__(self, input_shape):
        super(SaltDetectorProto, self).__init__()

        self.weights_init = keras.initializers.RandomNormal()

        self.features = Sequential([
            # pooling
            Conv2D(input_shape=input_shape, strides=(2, 2), filters=8, kernel_size=(3,3), activation='relu', padding='same'),
            Conv2D(filters=16, strides=(2, 2), kernel_size=(3,3), activation='relu', padding='same'),

            # bottleneck
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),

            # unsampling
            UpSampling2D(size=(2, 2)),
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
            UpSampling2D(size=(2, 2)),
            Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'),

            Conv2D(filters=1, kernel_size=(1,1))
        ])

    def call(self, inputs):
        x = self.features(inputs)
        return activations.sigmoid(x)

class SaltDetector(keras.Model):
    def __init__(self, input_shape):
        super(SaltDetector, self).__init__()

        self.weights_init = keras.initializers.RandomNormal()

        self.features = Sequential([
            # pooling
            Conv2D(input_shape=input_shape, strides=(2, 2), filters=8, kernel_size=(3,3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=16, strides=(2, 2), kernel_size=(3,3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=32, strides=(2, 2), kernel_size=(3,3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=64, strides=(2, 2), kernel_size=(3,3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=128, strides=(2, 2), kernel_size=(3,3), activation='relu', padding='same'),
            BatchNormalization(),

            # bottleneck
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),

            # unsampling
            UpSampling2D(size=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            UpSampling2D(size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            UpSampling2D(size=(2, 2)),
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            UpSampling2D(size=(2, 2)),
            Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            UpSampling2D(size=(2, 2)),
            Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),

            Conv2D(filters=1, kernel_size=(1,1))
        ])

    def call(self, inputs):
        x = self.features(inputs)
        return activations.sigmoid(x)

# constants
path_to_train_imgs = "./competition_data/train/images"
path_to_train_masks = "./competition_data/train/masks"
path_to_test_imgs = "./competition_data/test/images"

# load data
train_raw_imgs = loadImgsFromFolder(path_to_train_imgs)
train_raw_masks = loadImgsFromFolder(path_to_train_masks)
#test_raw_imgs = loadImgsFromFolder(path_to_test_imgs)

# prepare data
train_imgs = np.expand_dims(np.asarray(train_raw_imgs) / 255.0, axis=3)
#test_imgs = np.expand_dims(np.asarray(test_raw_imgs) / 255.0, axis=3)[0:2000]
train_masks = np.expand_dims(np.asarray(train_raw_masks) / 65535.0, axis=3)
(train_imgs, test_imgs, train_masks, test_masks) = train_test_split(train_imgs, train_masks, test_size=0.05)

print(train_imgs.shape[1:])
print(train_masks.shape)

# init model
model = SaltDetector(train_imgs.shape[1:])
optimizer = optimizers.Adam()
loss = losses.BinaryCrossentropy()

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('salt-iter4model.h5', save_best_only=True, verbose=1)
model.compile(loss=loss, optimizer=optimizer, metrics=[mean_iou])
#model.load_weights('salt-iter3model.h5')
model.fit(train_imgs, train_masks, validation_split=0.1, batch_size=10, epochs=35, callbacks=[checkpointer])

# predictions && evaluations on test data
preds_test = model.predict(test_imgs, verbose=1)
preds_test = (preds_test > 0.5).astype(np.uint8)
test_loss, test_iou = model.evaluate(test_imgs, test_masks)

# view imgs\masks\preds images 
cols = 3
rows = 6

for j in range(12):
    fig = plt.figure(figsize=(10,7))
    for i in range(j*rows,(j*rows)+rows):
        fig.add_subplot(rows, cols, ((i-j*rows)*3)+1)
        plt.imshow(np.dstack((test_imgs[i],test_imgs[i],test_imgs[i])))
        tmp = np.squeeze(test_masks[i]).astype(np.float32)
        fig.add_subplot(rows, cols, ((i-j*rows)*3)+2)
        plt.imshow(np.dstack((tmp,tmp,tmp)))
        tmp = np.squeeze(preds_test[i]).astype(np.float32)
        fig.add_subplot(rows, cols, ((i-j*rows)*3)+3)
        plt.imshow(np.dstack((tmp,tmp,tmp)))
    plt.show()
    plt.clf()
