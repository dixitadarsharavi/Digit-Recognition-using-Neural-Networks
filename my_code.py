#Using CNN

# example of loading the mnist dataset
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
# load dataset
#def load_dataset():
(trainX, trainy), (testX, testy) = mnist.load_data()
    # reshape dataset to have a single channel
trainX = trainX.reshape((trainX.shape[0], 28, 28,1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
# one hot encode target values
trainy =to_categorical(trainy)
testy =to_categorical(testy)
#    return trainX,trainy,testX,testy
# summarize loaded dataset
#print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
#print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
#for i in range(9):
#	# define subplot
#	pyplot.subplot(330+1+ i)
#	# plot raw pixel data
#	pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
## show the figure
#pyplot.show()
# scale pixels
#def prep_pixels(train, test):
	# convert from integers to floats
train_norm = trainX.astype('float32')
test_norm = testX.astype('float32')
	# normalize to range 0-1
train_norm = train_norm / 255.0
test_norm = test_norm / 255.0
	# return normalized images
#	return train_norm, test_norm

#Initializing the CNN
#def define_model():    
classifier= Sequential()
#1 Convolution+ ReLu
classifier.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu',kernel_initializer='he_uniform'))
#2 Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
#to increase accuracy
classifier.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2)))
#3 Flattening
classifier.add(Flatten())
# Full Connection
classifier.add(Dense(units=100,activation='relu',kernel_initializer='he_uniform'))#units= somewhere bw input and output nodes and since it is larger in our case we use somewhere around 100 (power of 2)
classifier.add(Dense(units=10,activation='softmax'))
# Compiling CNN
opt =SGD(lr=0.01, momentum=0.9)
classifier.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
#    return classifier
# run the test harness for evaluating a model
#def run_test_harness():
	# load dataset
	#trainX, trainy, testX, testy = load_dataset()
	# prepare pixel data
	#trainX, testX = prep_pixels(trainX, testX)
	# define model
	#classifier = define_model()
	# fit model
classifier.fit(trainX, trainy, epochs=10, batch_size=32, verbose=0)
acc = classifier.evaluate(testX, testy, verbose=0)
#print('> %.3f' % (acc * 100.0))
	# save model
	#classifier.save('final_model.h5')
# entry point, run the test harness
#run_test_harness()
img = load_img('sample_image.png', grayscale=True, target_size=(28, 28))
	# convert to array
img = img_to_array(img)
	# reshape into a single sample with 1 channel
img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
img = img.astype('float32')
img = img / 255.0
# predict the class
digit = classifier.predict_classes(img)
print(digit[0])