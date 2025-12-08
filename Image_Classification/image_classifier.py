import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,models,layers

##
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
##

(training_images,training_labels),(testing_images,testing_labels) = datasets.cifar10.load_data()
training_images,testing_images = training_images/255 , testing_images/255

class_name = ['Plane','Car','Bird','Cat','Deer','Dog','frog','Horse','Ship','Truck']

# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(training_images[i],cmap= plt.cm.binary)
#
#     plt.xlabel(class_name[training_labels[i][0]])
# plt.show()


training_images = training_images[:30000]
training_labels = training_labels[:30000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

#model = models.Sequential()
# model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3))) ##--> see feature
# model.add(layers.MaxPool2D(2,2))  # reduce to essential information
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.MaxPool2D(2,2))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64,activation='relu'))
# model.add(layers.Dense(10,activation='softmax'))
#
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#
# model.fit(training_images,training_labels,epochs=10,validation_data=(testing_images,testing_labels))
#
#
# loss , accuracy = model.evaluate(testing_images,testing_labels)
#
# print(f" loss {loss}")
# print(f" Accuracy { accuracy}")
#
# model.save('image_classification_model.keras')

model = models.load_model('image_classification_model.keras')

img = cv.imread('/Users/akash/Documents/neural Nine/Image_Classification/picture/car.jpg')
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

plt.imshow(img,cmap=plt.cm.binary)

prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)
print(f'Prediction is {class_name[index]}')

plt.show()

