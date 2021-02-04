#-*- coding: utf-8 -*-

import os
import datetime
from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np
from imgaug import augmenters as iaa
# import myModel
# from generator import PassengerFaceGenerator
# from data import getFaceData, getTrainDataset, getValAndTestDataset

#######################################################
# Set GPU
#######################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# GPU를 아예 못 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]=''
# GPU 0만 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]='0'
# GPU 1만 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]='0'
# GPU 0과 1을 보게 하려면: os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

#######################################################
# Set Memory
#######################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

#######################################################
# Set hyper parameters
#######################################################
input_shape = (254, 254, 3)
num_classes = 2
batch_size = 16
lr = 0.001
epoch = 5000

#######################################################
# Set parameters
#######################################################
data_dir = '/home/fsai3/mask_dataset/1st/Emergency/Mask'

m_name = 'efficientB7_02'

log_dir = os.path.join(
    "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-2nd',
)

#######################################################
# Generator function
#######################################################
class PassengerFaceGenerator(keras.utils.Sequence):
    def __init__(self, data_x, data_y, input_shape, num_classes, batch_size, augment=False, shuffle=False):

        self.data, self.bbox = self.divideData(data_x)
        self.label = data_y
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = None
        self.on_epoch_end()

    def divideData(self, data_x):
        # Divide data_x by jpg path and face bbox
        data_path = []
        data_bbox = []
        for data in data_x:
            data_path.append(data[0])
            data_bbox.append(data[1])

        return data_path, data_bbox

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size)) #데이터를 Batch_size만큼 나누기

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size] #범위 설정
        data = [self.data[i] for i in indexes] #인덱싱
        bbox =[self.bbox[i] for i in indexes]
        label = [self.label[i] for i in indexes]
        x, y = self.__data_gen(data, bbox, label)
        # print(x.shape, y.shape)
        return x, y

    def augmenter(self, images):
        seq = iaa.Sequential(
            [
                iaa.SomeOf((0, 5),
                           [
                               iaa.Identity(),
                               iaa.Rotate(),
                               # iaa.Posterize(),
                               # iaa.Sharpen(),
                               iaa.TranslateX(),
                               # iaa.GammaContrast(),
                               # iaa.Solarize(),
                               iaa.ShearX(),
                               iaa.TranslateY(),
                               # iaa.HistogramEqualization(),
                               # iaa.MultiplyHueAndSaturation(),
                               # iaa.MultiplyAndAddToBrightness(),
                               iaa.ShearY(),
                               iaa.ScaleX(),
                               iaa.ScaleY(),
                               iaa.Rot90(k=(1, 3))
                           ]
                           )
            ]
        )
        return seq.augment_images(images)

    def __data_gen(self, data, bbox, label):
        cv2.setNumThreads(0)
        batch_images = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                dtype=np.uint8)

        batch_class = np.zeros(shape=(self.batch_size, self.num_classes), dtype=np.float32)

        for i, img_path in enumerate(data):
            # Get img and Append data
            img = cv2.imread(img_path)
            xmin = int(bbox[i][0])
            ymin = int(bbox[i][1])
            xmax = int(bbox[i][2])
            ymax = int(bbox[i][3])
            img = img[ymin:ymax, xmin:xmax]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
            # img = img.astype(np.float32) / 255.
            batch_images[i] = img

            # Get label and Append data
            cls = label[i]
            cls = keras.utils.to_categorical(cls, num_classes=self.num_classes)
            batch_class[i] = cls

        # augment images
        if self.augment:
            batch_images = self.augmenter(batch_images)

        # Convert images data type and normalization
        batch_images = batch_images.astype(np.float32) / 255.

        return batch_images, batch_class


#######################################################
# Get generator
#######################################################
# train_generator = PassengerFaceGenerator(train_x, train_y, input_shape, num_classes, batch_size, augment=True, shuffle=True)
# val_generator = PassengerFaceGenerator(val_x, val_y, input_shape, num_classes, batch_size, augment=False, shuffle=False)

#######################################################
# Model function
#######################################################
def getEfficientB7(input_shape, num_classes):
    # Get EfficientB7 Model
    input_tensor = keras.layers.Input(shape=input_shape)
    backBone = keras.applications.EfficientNetB7(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # Top
    x = keras.layers.GlobalAveragePooling2D()(backBone.output)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_classes)(x)
    output = keras.layers.Activation('softmax')(x)

    return keras.Model(inputs=backBone.input, outputs=output)

#######################################################
# Get Model
#######################################################
# model = myModel.getResNet50(num_classes)
# model = myModel.getResNet152(num_classes)
# model = myModel.getDenseNet201(num_classes)
model = getEfficientB7(input_shape, num_classes)
model.summary()

#######################################################
# Compile Model
#######################################################
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

#######################################################
# Set callbacks
#######################################################
callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.5,
                                               patience=10,
                                               verbose=1,
                                               mode='min',
                                               min_lr=0.0001),
             keras.callbacks.ModelCheckpoint(filepath='./saved_models/2nd/' + m_name + '-{epoch:05d}.h5',
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='min',
                                             save_freq='epoch'),
             keras.callbacks.TensorBoard(log_dir=log_dir,
                                         profile_batch=0),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=50,
                                           verbose=1,
                                           mode='min')
             ]

#######################################################
# Train Model
#######################################################
model.fit(x=train_generator,
          epochs=epoch,
          verbose=1,
          callbacks=callbacks,
          validation_data=val_generator,
          max_queue_size=10,
          workers=4,
          use_multiprocessing=False
          )
