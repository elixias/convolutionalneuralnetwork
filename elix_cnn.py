from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np

#image augmentation (geometric)
#using flow from directory: https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#implementing callback to get validation accuracy after every epoch
from keras.callbacks import Callback
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = 0

    def on_epoch_end(self, epoch, logs={}):
        self.losses = logs.get('val_acc')

#classifier builder
def build_classifier():
    #init CNN
    classifier = Sequential()
    classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu'))#32 filters size 3x3 with input size of 3 channels 64x64
    #!tensorflow format!
    #input_shape(3,64,64) #theano format

    #pooling / reducing the conv maps
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    #adding more layers
    classifier.add(Convolution2D(32,3,3,activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))

    #flatten
    classifier.add(Flatten())

    #proceed to build ANN
    classifier.add(Dense(units=128,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=128,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier

#begin model fitting
kfolds = 2
no_train_data=8000
no_test_data=2000

accuracies = []
for x in range(0,kfolds):
    history=LossHistory()
    classifier = build_classifier()
    classifier.fit_generator(
            train_set,
            steps_per_epoch=no_train_data,
            epochs=1,
            validation_data=test_set,
            validation_steps=no_test_data,
            callbacks=[history])
    accuracies.append(history.losses)
print(accuracies)
accuracies = np.array(accuracies)
print("Mean:",accuracies.mean())
print("Std:",accuracies.std())

'''from keras.preprocessing import image
test_image = image.load_img('dataset/test_set/', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=4)#input has to be in a batch
results = classifier.predict(test_image)
print("Result:",results)
print("Classification index:",training_set.class_indices)'''