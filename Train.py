import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
import os



#csv files path
path = 'MURA-v1.1'
#csv files names
train_image_paths_csv = "train_image_paths.csv"
train_images_paths = pd.read_csv(os.path.join(path,train_image_paths_csv),dtype=str,header=None)
train_images_paths.columns = ['image_path']
train_images_paths['label'] = train_images_paths['image_path'].map(lambda x:'positive' if 'positive' in x else 'negative')
train_images_paths['category']  = train_images_paths['image_path'].apply(lambda x: x.split('/')[2])  
train_images_paths['patientId']  = train_images_paths['image_path'].apply(lambda x: x.split('/')[3].replace('patient',''))
total_number_of_training_images = np.shape(train_images_paths)[0]
print("total number of images:",total_number_of_training_images)
print ("number of null values", train_images_paths.isnull().sum())
print("number of training images:",np.shape(train_images_paths['image_path'])[0])
categories_counts = pd.DataFrame(train_images_paths['category'].value_counts())
print ('categories:\n',categories_counts )
print('number of patients:',train_images_paths['patientId'].nunique())
print('number of labels:',train_images_paths['label'].nunique())
print ('positive casses:',len(train_images_paths[train_images_paths['label']=='positive']))
print ('negative casses:',len(train_images_paths[train_images_paths['label']=='negative']))

Train_images=[]
Train_lbls=[]
from progressbar import ProgressBar
pbar = ProgressBar()

for i in pbar(range(0,total_number_of_training_images)):
    img=cv2.imread(train_images_paths.image_path[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)
    img = cv2.medianBlur(img, 3)
    Train_images.append(img)
    Train_lbls.append(train_images_paths.category[i]+'_'+train_images_paths.label[i])

def plot_images(images, title):
    nrows, ncols = 3, 3
    figsize = [2, 2]
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor=(1, 1, 1))
    for i, axi in enumerate(ax.flat):
        axi.imshow(images[i],cmap='gray')
        axi.set_axis_off()
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(pad=0.2, rect=[0, 0, 1, 0.9])
    plt.show()
    
print("Ploting Images")    
plot_images(Train_images[0:16], 'XR_SHOULDER')
plot_images(Train_images[8379:8395], 'XR_HUMERUS')
plot_images(Train_images[9651:9667], 'XR_FINGER')
plot_images(Train_images[14757:14773], 'XR_ELBOW')
plot_images(Train_images[19688:19704], 'XR_WRIST')
plot_images(Train_images[29440:29456], 'XR_FOREARM')
plot_images(Train_images[31265:31281], 'XR_HAND')

#data list to array convert
data1 = np.array(Train_images)
n_samples = len(data1)
data = data1.reshape((n_samples, -1))


#spilit Data
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(data, Train_lbls, test_size=0.25,random_state = 50) 


#SVM Training
from sklearn import svm
classifier = svm.SVC(decision_function_shape='ovo')
classifier.fit(X_train, y_train)
expected = y_test
predicted = classifier.predict(X_test)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
import pickle
pickle.dump(classifier, open('Model/SVM.sav', 'wb'))




#RF Training
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
expected = y_test
predicted = classifier.predict(X_test)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
import pickle
pickle.dump(classifier, open('Model/RF.sav', 'wb'))





#DT Training
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(X_train, y_train)
expected = y_test
predicted = classifier.predict(X_test)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
import pickle
pickle.dump(classifier, open('Model/DT.sav', 'wb'))




#KNN Training
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
expected = y_test
predicted = classifier.predict(X_test)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
import pickle
pickle.dump(classifier, open('Model/KNN.sav', 'wb'))





#Deep Learning
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
# data = pickle.load(open("Model/data.sav", 'rb'))
# Train_lbls = pickle.load(open("Model/lbls.sav", 'rb'))
# Train_images = pickle.load(open("Model/trimg.sav", 'rb'))
lbls=np.empty((len(data),1), dtype=list)
lbls=[]
for i in range(0,len(data)):
    if Train_lbls[i]=='XR_SHOULDER_positive' :
        lbls.append(0)
    if  Train_lbls[i]=='XR_SHOULDER_negative' :
        lbls.append(1)
    if Train_lbls[i]=='XR_HUMERUS_positive' :
        lbls.append(2)
    if  Train_lbls[i]=='XR_HUMERUS_negative' :
        lbls.append(3)
    if Train_lbls[i]=='XR_FINGER_positive' :
        lbls.append(4)
    if  Train_lbls[i]=='XR_FINGER_negative' :
        lbls.append(5)
    if Train_lbls[i]=='XR_ELBOW_positive' :
        lbls.append(6)
    if  Train_lbls[i]=='XR_ELBOW_negative' :
        lbls.append(7)
    if Train_lbls[i]=='XR_WRIST_positive' :
        lbls.append(8)
    if  Train_lbls[i]=='XR_WRIST_negative' :
        lbls.append(9)
    if Train_lbls[i]=='XR_FOREARM_positive' :
        lbls.append(10)
    if  Train_lbls[i]=='XR_FOREARM_negative' :
        lbls.append(11)
    if Train_lbls[i]=='XR_HAND_positive' :
        lbls.append(12)
    if  Train_lbls[i]=='XR_HAND_negative' :
        lbls.append(13)
data = np.array(Train_images)
labels = np.array(lbls) 
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.10,random_state = 50) 
#Reshape input data 
X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)
X_test = X_test.reshape(X_test.shape[0], 64, 64, 1)
#Normalize inputs from 0-255 to 0-1 
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
#One hot encoding of outputs
y_train = np_utils.to_categorical(y_train, num_classes=14)
y_test = np_utils.to_categorical(y_test, num_classes=14)
num_classes = y_test.shape[1]
#Build CNN model
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(64,64,1), activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (2,2), activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(14,activation="softmax"))
model.summary()
#Compile the model
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
#Fit the model
history = model.fit(X_train,y_train, epochs=15,batch_size=128,verbose=1) #50 epochs
scores = model.evaluate(X_test,y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
# Plot accuracy result
plt.plot(history.history['accuracy'])
plt.title('model accuracy plot')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('Model/plot1.png', bbox_inches='tight')
plt.show()
# Plot loss result
plt.plot(history.history['loss'])
plt.title('model loss plot')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('Model/plot2.png', bbox_inches='tight')
plt.show()
model.save("Model/model.h5")

