from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#init CNN
classifier = Sequential()
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu'))#32 filters size 3x3 with input size of 3 channels 64x64 !tensorflow format!
#input_shape(3,64,64) #theano format
