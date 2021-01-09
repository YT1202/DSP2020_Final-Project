# DSP2020_Final project

&nbsp;

### Outline
- Prerequisites
- Datasets
- Train
- Test

&nbsp;

## Prerequisites
- Win10
- Python 3.7.9
- CPU
- Keras 2.2.4
- numpy, pandas, sklearn, matplotlib, time
- ...

&nbsp;

## Datasets
The dataset for DSP2020 Final Project (258MB) : https://drive.google.com/file/d/1V7SFoR3G8rWufmru_bfhyqlrVONiKnED/view?usp=sharing. 

&nbsp;

## Training 
- The code of training is : 
```
Training.py
```

- When **Training.py** is successfully executed and completed, the **Pretrained-Model** can be obtained :
```
Model.h5
```

- Note:
  * Before training, you should ensure **Dataset** and **Training.py** are in the same path.
  * For training, if using CPU to train will spend at least 10000(s) because its epochs=500 and time~22s/epoch.
  * The size of model is **12.17MB** (<20MB)

&nbsp;

## Testing
- The code of training is : 
```
Testing.py
```

- When **Testing.py** is successfully executed and completed, the **Prediction result** can be obtained :
```
Submission_r08943166.csv
```

- Note:
  * Before testing, you should ensure **Dataset** , **Model.h5** and **Testing.py** are in the same path.
  * For testing, it will load dataset and preprocess it .
    
    Then, the pretrained-model will be loaded, and using it to do the validation and prediction.
    
    Finally, **Submission_r08943166.csv**(prediction result) will be obtained, and it conforms to the submited format.

&nbsp;

## Author
- Yi-Ting Wu

&nbsp;

## Reference
- Introduction of DSP, Fall 2020 : https://sites.google.com/view/dspfall2020
