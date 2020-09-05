#reading the training labels from the csv file
import pandas as pd
df = pd.read_csv ('C:\\Users\\Anuj Verma\\eclipse\\Downloads\\bonus-sml-2020\\SML_Train.csv') 
train_label= df.values[:,1]
train_label=train_label.astype("int")

#reading the data from the folders of train data and test data
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import glob
train_img_dir ="C:\\Users\\Anuj Verma\\eclipse\\Downloads\\bonus-sml-2020\\SML_Train"
test_img_dir ="C:\\Users\\Anuj Verma\\eclipse\\Downloads\\bonus-sml-2020\\SML_Test"
train_data_path =os.path.join(train_img_dir,'*g')
test_data_path = os.path.join(test_img_dir,'*g')
train_files = glob.glob(train_data_path)
test_files = glob.glob(test_data_path)


#traing image name
train_image_name =[]
for i in range (0,16000):
    x=train_files[i].split("\\")
    y=x[len(x)-1]
    train_image_name.append(y)

test_image_name_ = []
for i in range (0,1500):
    x=test_files[i].split("\\")
    y=x[len(x)-1]
    test_image_name_.append(y)

train_actual_label=[]
for i in range (0,16000):
    x = train_image_name[i].split(".")
    y = x[0].split("_")
    z = int(y[1])
    train_actual_label.append(train_label[z])
train_actual_label=np.array(train_actual_label)
train_actual_label=train_actual_label.astype("int")

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

#texture feature
import mahotas
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None,bins=8):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

#extracting features
train_data_set=[]
for i in range (0,16000):
    image = cv2.imread(train_files[i])
    #image = cv2.(image, None, 10, 10, 10, 20) 
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)
    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_hu_moments,fv_haralick,fv_histogram])
    train_data_set.append(global_feature)
train_data_set = np.array(train_data_set)
train_data_set.shape

#transformation
from sklearn.preprocessing import quantile_transform
final = quantile_transform(train_data_set,random_state=0,copy=True)

#train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        final, train_actual_label, train_size=0.75, test_size=0.25, random_state=42, shuffle=True)

#model for classification
from sklearn.ensemble import RandomForestClassifier as RFC
rfc_b = RFC(n_estimators=300,max_depth=20)
rfc_b.fit(final,train_actual_label)

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
# KFold Cross Validation approach
kf = KFold(n_splits=5,shuffle=False)
kf.split(train_data_set)    
# Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
accuracy_model = []

for train_index, test_index in kf.split(train_data_set):
    # Split train-test
    X_train, X_test = train_data_set[train_index], train_data_set[test_index]
    y_train, y_test = train_actual_label[train_index], train_actual_label[test_index]
    # Train the model
    
    model = rfc_b.fit(X_train, y_train)
    # Append to accuracy_model the accuracy of the model
    accuracy_model.append(accuracy_score(y_test, model.predict(X_test), normalize=True)*100)

#preparing for the test
test_data_set=[]
for i in range (0,1500):
    image=cv2.imread(test_files[i])
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)
    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_hu_moments,fv_haralick,fv_histogram])
    test_data_set.append(global_feature)
test_data_set = np.array(test_data_set)
test_data_set.shape


#transform
final_test = quantile_transform(test_data_set,random_state=0,copy=True)
pred=rfc_b.predict(final_test)

ans_arr=np.zeros((1500,))
for i in range(0,1500):
    x = test_image_name_[i].split(".")
    y = x[0].split("_")
    z = int(y[1])
    ans_arr[z]=pred[i]
ans_arr=ans_arr.astype("int")
ans_arr

name_arr=[]
for i in range(0,1500):
    x="Test"+"_"+str(i)+"."+"jpg"
    name_arr.append(x)
name_arr=np.array(name_arr)
name_arr.shape


arr =[]
for i in range(1500):
    l =[name_arr[i],ans_arr[i]]
    arr.append(l)


df = pd.DataFrame(arr, columns=['id','category'])
df.to_csv('C:\\Users\\Anuj Verma\\eclipse\\Downloads\\2017026_Anuj_submission.csv', index=False)