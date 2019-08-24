# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 21:46:47 2019

@author: 88696
"""
'''
改良Mnist_MLP_save_model,每次訓練累積權重(weight)
'''

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

def show_image(image):
    fig = plt.gcf() # get current figure
    fig.set_size_inches(2,2) 
    plt.imshow(image, cmap='binary') # 黑白灰階顯示
    plt.show()
    
def show_predictions(images, labels, predictions, start_id, num):
    plt.gcf().set_size_inches(12,14)
    if num>25:  num = 25    
    for i in range(num):
        ax = plt.subplot(5, 5, i+1)
        ax.imshow(images[start_id], cmap='binary')      
        if(len(predictions) > 0):
            title = 'ai = ' + str(predictions[i])
            title += ("(O)" if predictions[i] == labels[i] else "(X)")
            title += "\nlabel = " + str(labels[i])
        else:
            title = "label = " + str(labels[i])      
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        start_id += 1
    plt.show()

# pre-process
(train_feature, train_label),\
(test_feature, test_label) = mnist.load_data()

train_feature_vector = train_feature.reshape(len(train_feature), train_feature.shape[1]*train_feature.shape[2]).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature), test_feature.shape[1]*test_feature.shape[2]).astype('float32') 

train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255

train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)

# building model
model = Sequential()
model.add(Dense(units=256, 
                input_dim=784, 
                kernel_initializer='normal',
                activation='relu'))
model.add(Dense(units=10, 
                kernel_initializer='normal',
                activation='softmax'))

# defining the training method
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

### ---new---
# load the parameters 
try:
    model.load_weights("Mnist_MLP_model_v2.weight")
    print("載入參數成功! 繼續訓練模型...")
except:
    print("載入參數失敗，開始訓練新模型...")

# training model
train_history =  model.fit(x=train_feature_normalize, 
                  y=train_label_onehot,
                  validation_split=0.2,
                  epochs=1, ### ---new---
                  batch_size=200,
                  verbose=2 )
# evaluate 
scores = model.evaluate(test_feature_normalize, test_label_onehot)
print("準確率:", scores[1])
### ---new---
num = len(open('accuracy_record.txt','r').readlines())
with open('accuracy_record.txt','a') as fo:
    fo.write("\n第"+str(num+1)+"次訓練結果: 準確率="+str(scores[1]))
print("本次準確率紀錄儲存完畢!")
    
# predict
my_pred = model.predict_classes(test_feature_normalize)

#show
show_predictions(images = test_feature, labels = test_label,
                 predictions = my_pred, start_id = 0, num = 10)

#save 
model.save('Mnist_MLP_model_v2.h5')
print("Mnist_MLP_model.h5模型儲存完畢!")
model.save('Mnist_MLP_model_v2.weight')
print("Mnist_MLP_model.h5模型參數儲存完畢!")

del model