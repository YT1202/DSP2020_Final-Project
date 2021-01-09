import time
import sklearn
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten
from keras.layers import Dropout


np.random.seed(10)


# 計時開始
tStart = time.time()


# =============================================================================
# Step_1  載入資料集  
x_train, y_train = np.load("traindata.npy"), np.load("trainlabel.npy")
x_val, y_val = np.load("validationdata.npy"), np.load("validationlabel.npy")
x_test = np.load("testdata.npy")
print("x_train size: ", x_train.shape, "\n-------------------------------------")
print("x_val size: ",x_val.shape, "\n-------------------------------------")
print("x_test size: ", x_test.shape, "\n-------------------------------------")
print("y_train size: ", x_train.shape, "\n-------------------------------------")
print("y_val size: ",x_val.shape, "\n=======================================")
# =============================================================================


# =============================================================================
# Step_2  資料預處理
x_train_0 = StandardScaler().fit_transform(x_train[:, 0, :])
x_train_1 = StandardScaler().fit_transform(x_train[:, 1, :])
x_train = np.hstack([x_train_0, x_train_1]).reshape(3360, 2, 20000)

x_val_0 = StandardScaler().fit_transform(x_val[:, 0, :])
x_val_1 = StandardScaler().fit_transform(x_val[:, 1, :])
x_val = np.hstack([x_val_0, x_val_1]).reshape(480, 2, 20000)

x_test_0 = StandardScaler().fit_transform(x_test[:, 0, :])
x_test_1 = StandardScaler().fit_transform(x_test[:, 1, :])
x_test = np.hstack([x_test_0, x_test_1]).reshape(960, 2, 20000)
# =============================================================================


# =============================================================================
# Step_3  利用OneHot對label進行編碼
y_train_onehot = np_utils.to_categorical(y_train)
y_val_onehot = np_utils.to_categorical(y_val)
# =============================================================================


# =============================================================================
# Step_4  建立訓練模型
model = Sequential()
# input (None, 20000) / output (6, 10000)
model.add(Conv1D(batch_input_shape=(None, 2, 20000),
                 filters=6,
                 kernel_size=5,
                 strides=1,
                 padding='same',     
                 data_format='channels_first',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2,
                       strides=2,
                       padding='same',
                       data_format='channels_first'))
# input (6, 10000) / output (6, 5000)
model.add(Conv1D(filters=6,
                 kernel_size=5,
                 strides=1,
                 padding='same',     
                 data_format='channels_first',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2,
                       strides=2,
                       padding='same',
                       data_format='channels_first'))
# input (6, 5000) / output (16, 2500)
model.add(Conv1D(filters=16,
                 kernel_size=5,
                 strides=1,
                 padding='same',     
                 data_format='channels_first',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2,
                       strides=2,
                       padding='same',
                       data_format='channels_first'))
# input (16, 2500) / output (16, 1250)
model.add(Conv1D(filters=16,
                 kernel_size=5,
                 strides=1,
                 padding='same',     
                 data_format='channels_first',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2,
                       strides=2,
                       padding='same',
                       data_format='channels_first'))
# input (16, 1250) / output (36, 625)
model.add(Conv1D(filters=36,
                 kernel_size=5,
                 strides=1,
                 padding='same',     
                 data_format='channels_first',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2,
                       strides=2,
                       padding='same',
                       data_format='channels_first'))
# input (36, 625) / output (36, 313)
model.add(Conv1D(filters=36,
                 kernel_size=5,
                 strides=1,
                 padding='same',     
                 data_format='channels_first',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2,
                       strides=2,
                       padding='same',
                       data_format='channels_first'))
# input (36, 625) / output (64, 157)
model.add(Conv1D(filters=64,
                 kernel_size=5,
                 strides=1,
                 padding='same',     
                 data_format='channels_first',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2,
                       strides=2,
                       padding='same',
                       data_format='channels_first'))
# input (64, 157) / output (64, 79)
model.add(Conv1D(filters=64,
                 kernel_size=5,
                 strides=1,
                 padding='same',     
                 data_format='channels_first',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2,
                       strides=2,
                       padding='same',
                       data_format='channels_first'))
# input (64, 79) / output (108, 40)
model.add(Conv1D(filters=108,
                 kernel_size=5,
                 strides=1,
                 padding='same',     
                 data_format='channels_first',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2,
                       strides=2,
                       padding='same',
                       data_format='channels_first'))
# input (108, 40) / output (108, 20)
model.add(Conv1D(filters=108,
                 kernel_size=5,
                 strides=1,
                 padding='same',     
                 data_format='channels_first',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2,
                       strides=2,
                       padding='same',
                       data_format='channels_first'))
# input (108, 20) / output (156, 10)
model.add(Conv1D(filters=156,
                 kernel_size=5,
                 strides=1,
                 padding='same',     
                 data_format='channels_first',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2,
                       strides=2,
                       padding='same',
                       data_format='channels_first'))
# input (156, 10) / output (156, 5)
model.add(Conv1D(filters=156,
                 kernel_size=5,
                 strides=1,
                 padding='same',     
                 data_format='channels_first',
                 activation='relu'))
model.add(MaxPooling1D(pool_size=2,
                       strides=2,
                       padding='same',
                       data_format='channels_first'))
model.add(Dropout(0.25))
# Fc
model.add(Flatten())
model.add(Dense(units=780, 
                kernel_initializer='random_uniform',
                bias_initializer='zeros',
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=128, 
                kernel_initializer='random_uniform',
                bias_initializer='zeros',
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=4,  
                activation='softmax'))

print(model.summary())
# =============================================================================


# =============================================================================
# Step_5  訓練參數設定
keras.optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history=model.fit(x=x_train, 
                        y=y_train_onehot,
                        validation_data=(x_val, y_val_onehot), 
                        epochs=500, 
                        batch_size=64,
                        shuffle=True,
                        verbose=1)
# =============================================================================


# =============================================================================
# Step_6  顯示訓練結果: Accurancy & Loss
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='right')
    plt.show()

show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
# =============================================================================


# =============================================================================
# Step_7  用測試集檢驗模型
scores = model.evaluate(x_val, y_val_onehot)
print('scores= ', scores[0])
print('accurancy= ', scores[1])
# =============================================================================


# =============================================================================
# Step_8   儲存模型
model.save('model.h5')
# =============================================================================


# 計時結束
tEnd = time.time()
print('==========================================\ntime : ', tEnd - tStart, '(s)')
