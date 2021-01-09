import keras
import sklearn
import pandas as pd
import numpy as np 
from keras.utils import np_utils
from keras.models import load_model
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Step_1 載入Valadtion & Test資料集
x_val, y_val = np.load("validationdata.npy"), np.load("validationlabel.npy") 
x_test = np.load("testdata.npy")
print("x_val size: ", x_val.shape, "\n-------------------------------------")
print("x_test size: ", x_test.shape, "\n-------------------------------------")
print("y_val size: ", y_val.shape, "\n=======================================")
# =============================================================================


# =============================================================================
# Step_2  利用OneHot對label進行編碼 
y_val_onehot = np_utils.to_categorical(y_val)
# =============================================================================


# =============================================================================
# Step_3 對數據進行預處理
x_val_0 = StandardScaler().fit_transform(x_val[:, 0, :])
x_val_1 = StandardScaler().fit_transform(x_val[:, 1, :])
x_val = np.hstack([x_val_0, x_val_1]).reshape(480, 2, 20000)

x_test_0 = StandardScaler().fit_transform(x_test[:, 0, :])
x_test_1 = StandardScaler().fit_transform(x_test[:, 1, :])
x_test = np.hstack([x_test_0, x_test_1]).reshape(960, 2, 20000)
# =============================================================================


# =============================================================================
# Step_4  載入訓練好的模型
model = load_model('model.h5')
print(model.summary(), "\n\n=======================================")
# =============================================================================


# =============================================================================
# Step_5  用該模型進行預測
prediction = model.predict_classes(x_test)
y_test = prediction.reshape(-1,1)
# =============================================================================


# =============================================================================
# Step_6  產生Submit的格式且儲存 
id_test = np.array(range(0, 960)).reshape(-1,1)
submission = np.hstack([id_test, y_test])
submission = pd.DataFrame(submission, columns=["id", "category"])
submission.to_csv('submission.csv', sep=',', index=False)
# =============================================================================
