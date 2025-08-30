# Age, gender and expression recognition application

## 1. Various sources
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


