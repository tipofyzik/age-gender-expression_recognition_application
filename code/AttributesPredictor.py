from PIL import Image
import numpy as np
import logging
import traceback

# tflite fallback
try:
    import tflite_runtime.interpreter as tflite
except ImportError as e:
    raise ImportError("tflite_runtime не установлен. Убедитесь, что он указан в requirements в buildozer.spec") from e

import sys
sys.path.insert(0, './libs')
from fdlite import FaceDetection, FaceDetectionModel



# Логгер
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



class AttributesPredictor:
    def __init__(self):
        try:
            self.age_interpreter = tflite.Interpreter(model_path="models/MobileNet_Age.tflite")
            self.age_interpreter.allocate_tensors()

            self.gender_interpreter = tflite.Interpreter(model_path="models/MobileNet_Gender.tflite")
            self.gender_interpreter.allocate_tensors()

            self.emotion_interpreter = tflite.Interpreter(model_path="models/emotion_model.tflite")
            self.emotion_interpreter.allocate_tensors()

            self.face_interpreter = FaceDetection(model_type=FaceDetectionModel.FRONT_CAMERA, model_path="models/")
        except Exception as e:
            logger.error("Failed to initialize interpreters or face detector", exc_info=True)
            raise

    def preprocess_face(self, face_roi, target_size):
        try:
            face_img = face_roi.resize(target_size)
            face_array = np.array(face_img).astype('float32') / 255.0
            face_array = np.expand_dims(face_array, axis=0)
            return face_array
        except Exception as e:
            logger.error("Error in preprocess_face", exc_info=True)
            raise

    def detect_faces(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            img_rgb = np.array(image)
            detections = self.face_interpreter(img_rgb)

            faces = []
            w, h = image.size
            for det in detections:
                xmin = int(det.bbox.xmin * w)
                ymin = int(det.bbox.ymin * h)
                xmax = int(det.bbox.xmax * w)
                ymax = int(det.bbox.ymax * h)

                width = xmax - xmin
                height = ymax - ymin
                if width > 0 and height > 0:
                    faces.append((xmin, ymin, width, height))

            return faces, image
        except Exception as e:
            logger.error("Error in detect_faces", exc_info=True)
            raise

    def analyze_image(self, image_path):
        results = []
        try:
            faces, image = self.detect_faces(image_path)
        except Exception as e:
            return [{"error": f"Face detection failed: {e}"}]

        if not faces:
            return [{"error": "No faces detected"}]

        for (x, y, w, h) in faces:
            try:
                face_roi = image.crop((x, y, x + w, y + h))
                if face_roi.size[0] == 0 or face_roi.size[1] == 0:
                    continue

                age_input = self.preprocess_face(face_roi, target_size=(224, 224))
                gender_input = self.preprocess_face(face_roi, target_size=(224, 224))
                emotion_input = self.preprocess_face(face_roi, target_size=(64, 64))

                age = self.predict_age(age_input)
                gender = self.predict_gender(gender_input)
                emotion = self.predict_emotion(emotion_input)

                results.append({
                    "age": age,
                    "gender": gender,
                    "emotion": emotion,
                    "face_box": (x, y, w, h)
                })
            except Exception as e:
                logger.error("Error while analyzing a face", exc_info=True)
                results.append({"error": f"Failed to analyze face: {e}"})

        return results

    def predict_age(self, face_data):
        try:
            input_details = self.age_interpreter.get_input_details()
            output_details = self.age_interpreter.get_output_details()

            self.age_interpreter.set_tensor(input_details[0]['index'], face_data.astype(np.float32))
            self.age_interpreter.invoke()
            output_data = self.age_interpreter.get_tensor(output_details[0]['index'])

            age_ranges = [
                "4 - 6 years old",
                "7 - 8 years old",
                "9 - 11 years old",
                "12 - 19 years old",
                "20 - 27 years old",
                "28 - 35 years old",
                "36 - 45 years old",
                "46 - 60 years old",
                "61 - 75 years old"
            ]
            age_idx = np.argmax(output_data)
            return age_ranges[age_idx]
        except Exception as e:
            logger.error("Error in predict_age", exc_info=True)
            raise

    def predict_gender(self, face_data):
        try:
            input_details = self.gender_interpreter.get_input_details()
            output_details = self.gender_interpreter.get_output_details()

            self.gender_interpreter.set_tensor(input_details[0]['index'], face_data.astype(np.float32))
            self.gender_interpreter.invoke()
            output_data = self.gender_interpreter.get_tensor(output_details[0]['index'])

            genders = ["Female", "Male"]
            genders_idx = np.argmax(output_data)
            return genders[genders_idx]
        except Exception as e:
            logger.error("Error in predict_gender", exc_info=True)
            raise

    def predict_emotion(self, face_data):
        try:
            input_details = self.emotion_interpreter.get_input_details()
            output_details = self.emotion_interpreter.get_output_details()

            self.emotion_interpreter.set_tensor(input_details[0]['index'], face_data.astype(np.float32))
            self.emotion_interpreter.invoke()
            prediction = self.emotion_interpreter.get_tensor(output_details[0]['index'])[0]

            emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
            emotion_idx = np.argmax(prediction)
            return emotions[emotion_idx]
        except Exception as e:
            logger.error("Error in predict_emotion", exc_info=True)
            raise
