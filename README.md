# Age, gender and expression recognition application

## 1. Various sources
### Face-detection-tflite
In this project, face-detection-tflite library developed by patlevin was used: https://github.com/patlevin/face-detection-tflite  

In the **face_detection.py**, **face_landmark.py**, **iris_landmark.py** files some fixes for the project purposes been made. Namely, I replaced  
```python
import tensorflow as tf
# Other code
self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
```  
with  
```python
try:
    import tflite_runtime.interpreter as tflite
except ImportError as e:
    import tensorflow.lite as tflite
# Other code
self.interpreter = tflite.Interpreter(model_path=self.model_path)
```
This is done in order to run my mobile application with the **_tflite_runtime_** library but not with the full tensorflow package. In addition, it allowed running application on both pc and mobile device, that came nandy during testing and evaluation phases.

### Pretrained models
1. Face-detection-tflite for face detection via a camera (front camera only in this case): https://github.com/patlevin/face-detection-tflite
2. MobileNet for age and gender recognition: https://github.com/radualexandrub/Age-Gender-Classification-on-RaspberryPi4-with-TFLite-PyQt5
3. FER model for emotion recognition: https://github.com/vicksam/fer-model

### Datasets for performance evaluation
1. Gender dataset, folder "Test": https://www.kaggle.com/datasets/rashikrahmanpritom/gender-recognition-dataset
2. Age dataset, should be adjusted via the **PerformanceEvaluator.py** script: https://www.kaggle.com/datasets/frabbisw/facial-age
3. Emotion dataset: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

## 2.Evaluation
To evaluate model performance on both a pc and an edge-device, download the main_eval.py, PerformanceEvaluator.py, and TransformPhoneResultsToMetrics.py files. Rename main_eval.py to main.py (remove the original main.py to the other directory). And build an application with this new main file. Also, download testing datasets from here and unpack them into the download folder on your phone. Dataset can be accessed via the link: https://drive.google.com/drive/folders/1TLMC87HDCJP2mMYNK2AgfMpv3lNRG_M8?usp=drive_link  
