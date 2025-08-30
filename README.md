# Age, gender and expression recognition application

Library source: https://github.com/patlevin/face-detection-tflite
Replaced import for the project purposes  
```python
import tensorflow as tf

```python
try:
    import tflite_runtime.interpreter as tflite
except ImportError as e:
    import tensorflow.lite as tflite
