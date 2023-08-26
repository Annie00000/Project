import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model

from google.colab import drive
drive.mount('/content/drive')


# load data
data_path = '/content/drive/MyDrive/Colab Notebooks/AOI/aoi'
train_list = pd.read_csv(os.path.join(data_path, 'train.csv'), index_col=False)
train_list.head()


# Check image imformation
data_path = '/content/drive/MyDrive/Colab Notebooks/AOI/aoi/train_images/'
img = cv2.imread(os.path.join(data_path, train_list.loc[0,'ID']))
print(f"image shape: {img.shape}")
print(f"data type: {img.dtype}")
print(f"min: {img.min()}, max: {img.max()}")  # 像素值
plt.imshow(img)
plt.show()


# View images of each class
normal_list = train_list[train_list['Label']==0]['ID'].values
void_list = train_list[train_list['Label']==1]['ID'].values
horizontal_defect_list = train_list[train_list['Label']==2]['ID'].values
vertical_defect_list = train_list[train_list['Label']==3]['ID'].values
edge_defect_list = train_list[train_list['Label']==4]['ID'].values
particle_list = train_list[train_list['Label']==5]['ID'].values

label=[normal_list,void_list,horizontal_defect_list,vertical_defect_list,edge_defect_list,particle_list]

defect = ['normal','void','horizontal defect','vertical defect','edge defect','particle']

plt.figure(figsize=(12, 6))
for i in range(6):
  plt.subplot(2, 3, i+1)
  img = cv2.imread(os.path.join(data_path, label[i][i]),0)
  plt.imshow(img,cmap='gray')
  plt.axis("off")
  plt.title(defect[i])
plt.suptitle(f"Variety of samples", fontsize=12)
plt.show()


# Check the distribution of label
train_list['Label'].value_counts()
sns.countplot(x='Label', data=train_list)





# Train validation data split
train, valid = train_test_split(train_list, test_size=0.2, random_state=42)

train.reset_index(drop=True, inplace=True)
valid.reset_index(drop=True, inplace=True)

train['Label']=train['Label'].astype('str')
valid['Label']=valid['Label'].astype('str')



##  Dealing with data imbalance : adjust weight
unique, counts = np.unique(train['Label'].values , return_counts=True)
print('unique ', unique)
print('counts: ', counts)

# Increase the penalty weight for data imbalance
class_weights = compute_class_weight(class_weight='balanced', classes= unique, y = train['Label'])
class_weights = {i:value for i, value in enumerate(class_weights)}
print(class_weights)



# Data Augmentation
train_datagen = ImageDataGenerator(
         horizontal_flip = True,
         vertical_flip = True,
         width_shift_range = 0.05,
         height_shift_range = 0.05,
         preprocessing_function = preprocess_input)

valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)


# img_shape should be set according to the input limit of the model (Xception)
img_shape = (299,299)
batch_size = 16

train_generator = train_datagen.flow_from_dataframe(dataframe=train,
                            directory= data_path,
                            x_col='ID',
                            y_col='Label',
                            target_size=img_shape,
                            batch_size=batch_size,
                            class_mode='categorical',
                            shuffle=True)

valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid,
                            directory= data_path,
                            x_col='ID',
                            y_col='Label',
                            target_size=img_shape,
                            batch_size=batch_size,
                            class_mode='categorical',
                            shuffle=True)






## Modeling
# Load the pre-trained model densenet, apply the weight of imagenet
model = Xception(include_top=False, weights='imagenet', input_tensor=Input(shape=(299,299,3))) #include_top:是否包括頂層的全連接層

# Define the output layer, update the number of categories of the output
x = GlobalAveragePooling2D()(model.output)
predictions = Dense(6, activation='softmax')(x)
model = Model(inputs = model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
       loss='categorical_crossentropy',
       metrics=['accuracy'])



## Callback 
# Set the model training mechanism : ModelCheckpoint / dynamic learning rate / Earlystop
filepath = 'Xception_best.h5'
checkpoint = ModelCheckpoint(filepath, verbose=1,
                monitor='val_accuracy', mode='max', save_best_only=True)

reduce_learning_rate= ReduceLROnPlateau(monitor='val_loss',
                    factor=0.5,  # 縮放學習率的值
                    patience=5,  # 5 epochs 內loss沒下降就要調整LR
                    mode='min',
                    verbose=1,  # 信息展示模式
                    min_learning_rate=1e-5)

early_stop = EarlyStopping(monitor='val_loss', # 或 val_acc 验证集的正确率
              patience=10, # 可以忍受在多少个epoch内没有改进？
              mode='min',
              verbose=1)

callbacks_list = [checkpoint, early_stop, reduce_learning_rate]



# Set the number of pictures for each epoch training
def num_steps_per_epoch(data_generator, batch_size):
    if data_generator.n % batch_size==0:
        return data_generator.n//batch_size
    else:
        return data_generator.n//batch_size + 1

steps_per_epoch_train = num_steps_per_epoch(train_generator, batch_size)
steps_per_epoch_val = num_steps_per_epoch(valid_generator, batch_size)




# Training
history = model.fit_generator(train_generator,
                steps_per_epoch=steps_per_epoch_train,
                epochs=50,
                validation_data=valid_generator,
                validation_steps = steps_per_epoch_val,
                class_weight = class_weights,
                callbacks = callbacks_list)


# Accuracy and loss for training and validation
def plot_performance(epochs, history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

epochs = history.epoch
plot_performance(epochs, history)





# Testing data prediction
test_list = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/AOI/aoi/test.csv',index_col=False)
test_list['Label'] = '7'

test_imagepath = '/content/drive/My Drive/Colab Notebooks/AOI/aoi/test_images/'

test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
test_generator = test_datagen.flow_from_dataframe( dataframe=test_list,
                directory = test_imagepath,
                x_col="ID",
                y_col="Label",
                target_size=img_shape,
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False)
test_steps = len(test_generator)


# The prediction will be a probability, so it needs to be converted
y_test_predprob = model.predict(test_generator, steps = test_steps)
y_test_pred = y_test_predprob.argmax(-1)
y_test_pred

test_list['Label'] = y_test_pred
test_list.to_csv('/content/drive/MyDrive/Colab Notebooks/AOI/aoi/Xception_submit.csv',index=False)
