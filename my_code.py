# %-----------------------------------------------------------------------%

# Date: 15-October-2019
# Author: Adarsha Dixit

# %-----------------------------------------------------------------------%

# Importing MNIST  dataset
from keras.datasets import mnist

#from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.models import load_model

# Load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()

# Re-shape dataset to have a single color channel
trainX = trainX.reshape((trainX.shape[0], 28, 28,1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

# One hot encode target values (Categorise based on the unique classes)
trainy =to_categorical(trainy)
testy =to_categorical(testy)

# convert from integers to floats
train_norm = trainX.astype('float32')
test_norm = testX.astype('float32')

# normalize to range 0-1
train_norm = train_norm / 255.0
test_norm = test_norm / 255.0

# Initializing the CNN
#def define_model():    
classifier= Sequential()

# 1 Convolution+ ReLu
classifier.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu',kernel_initializer='he_uniform'))

# 2 Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# To increase accuracy
classifier.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2)))

# 3 Flattening
classifier.add(Flatten())

# Full Connection
# units= somewhere bw input and output nodes and since it is larger in our case we use somewhere around 100 (power of 2)
classifier.add(Dense(units=100,activation='relu',kernel_initializer='he_uniform'))
classifier.add(Dense(units=10,activation='softmax'))

# Compiling CNN
opt =SGD(lr=0.01, momentum=0.9)
classifier.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

classifier.fit(trainX, trainy, epochs=10, batch_size=32, verbose=0)
acc = classifier.evaluate(testX, testy, verbose=0)

img = load_img('sample_image.png', grayscale=True, target_size=(28, 28))

# Convert to array
img = img_to_array(img)

# Reshape into a single sample with 1 channel
img = img.reshape(1, 28, 28, 1)

# Prepare pixel data
img = img.astype('float32')
img = img / 255.0

# Predict the class
digit = classifier.predict_classes(img)
print(digit[0])