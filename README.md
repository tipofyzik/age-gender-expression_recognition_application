# Age, gender and expression recognition application

## 1. Various sources
In this project, face-detection-tflite library developed by patlevin was used: https://github.com/patlevin/face-detection-tflite  

Import statements in the **face_detection.py**, **face_landmark.py**, **iris_landmark.py** files have been replaced for the project purposes:  
This was replaced
```python
import tensorflow as tf
```  
with  
```python
try:
    import tflite_runtime.interpreter as tflite
except ImportError as e:
    import tensorflow.lite as tflite
```

In addition, the line  
```python
self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
```
was replaced with 
```python
self.interpreter = tflite.Interpreter(model_path=self.model_path)
```
