from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils

def show_image(image):
    fig = plt.gcf() # get current figure
    fig.set_size_inches(2,2) 
    plt.imshow(image, cmap='binary') # 黑白灰階顯示
    plt.show()
    
def show_images_labels_predictions(images, labels, predictions, start_id, num=10):
    plt.gcf().set_size_inches(12,14)
    if num>25:  num = 25    
    for i in range(num):
        ax = plt.subplot(5, 5, i+1)
        #顯示黑白圖片
        ax.imshow(images[start_id], cmap='binary')
        
        #有 AI 預測結果資料, 才在標題顯示預測結果
        if(len(predictions) > 0):
            title = 'ai = ' + str(predictions[i])
            title += ("(O)" if predictions[i] == labels[i] else "(X)")
            title += "\nlabel = " + str(labels[i])
        #沒有 AI 預測結果資料, 只在標題顯示真實數值
        else:
            title = "label = " + str(labels[i])
        
        #X,Y軸不顯示刻度
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([]);
        start_id += 1
    plt.show()

#######################
# 一.Feature 預處理  ##
######################
#===========================================================
# 1.建立訓練資料和測試資料
#===========================================================
(train_feature, train_label), \
(test_feature, test_label) = mnist.load_data()

'''
print("# of training features: %d"%(len(train_feature))) #60000
print("# of training labels:   %d"%(len(train_label)))   #60000
print("# of testing  features: %d"%(len(test_feature)))  #10000
print("# of testing  labels:   %d"%(len(test_label)))    #10000
      
print("Dimension of training features: ",train_feature.shape) #(60000, 28, 28)
print("Dimension of training labels:   ",train_label.shape)   #(60000,)
print("Dimension of testing  features: ",test_feature.shape)  #(10000, 28, 28)
print("Dimension of testing  labels:   ",test_label.shape)    #(10000,)
'''   
#show_image(train_feature[0])
#show_images_labels_predictions(train_feature, train_label, [], 0, 10)

#===========================================================
# 2.將 Features 特徵值換為 784個 float 數字的 1 維向量
#===========================================================

w = train_feature.shape[1]; h = train_feature.shape[2] # w = h = 28
train_feature_vector = train_feature.reshape(len(train_feature), w * h).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature), w * h).astype('float32')
#print(train_feature_vector.shape, test_feature_vector.shape, sep='\n') #(60000, 784) ; (10000, 784)
#print(train_feature[0])
#print(train_feature_vector[0]) 

#===========================================================
# 3. Features 特徵值標準化
#===========================================================
train_feature_normalize = train_feature_vector/255
test_feature_normalize = test_feature_vector/255
#print(train_feature_normalize[0]) 


#######################
# 二.Feature 預處理  ##
######################
#print(train_label[:5])
train_label_one_hot = np_utils.to_categorical(train_label)
test_label_one_hot = np_utils.to_categorical(test_label)
#print(train_label_one_hot[:5])