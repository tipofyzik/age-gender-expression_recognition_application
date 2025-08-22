from PIL import Image
import numpy as np

# Tensorflow-light for mobile app
try:
    import tflite_runtime.interpreter as tflite
except ImportError as e:
    raise ImportError("tflite_runtime lib is not installed.") from e

"""Manually importing library for face detection to work. Library has custom fixes so it suits the project goals.
Source: https://github.com/patlevin/face-detection-tflite
"""
import sys
sys.path.insert(0, './libs')
from fdlite import FaceDetection, FaceDetectionModel



# Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



class AttributesPredictor:
    """
    A class for predicting face attributes (age, gender, emotions).
    Uses TFLite models and a custom face detector.
    """
    def __init__(self) -> None:
        """
        Initializes TFLite interpreters for age, gender, and emotion models,
        and loads the face detection model.
        
        Raises:
            Exception: If any of the interpreters or the face detector cannot be initialized.
        """
        try:
            self.age_interpreter = tflite.Interpreter(model_path="models/MobileNet_Age.tflite")
            self.age_interpreter.allocate_tensors()

            self.gender_interpreter = tflite.Interpreter(model_path="models/MobileNet_Gender.tflite")
            self.gender_interpreter.allocate_tensors()

            self.emotion_interpreter = tflite.Interpreter(model_path="models/emotion_model.tflite")
            self.emotion_interpreter.allocate_tensors()

            self.face_interpreter = FaceDetection(model_type=FaceDetectionModel.FRONT_CAMERA, model_path="models/")

            self.last_cropped_face = None
        except Exception:
            logger.error("Failed to initialize interpreters or face detector", exc_info=True)
            raise
        
    def preprocess_face(self, face_roi: Image.Image, 
                        target_size: tuple[int, int]) -> np.ndarray:
        """
        Preprocesses a face image for model input:
        resizes, crops, and normalizes it.

        Args:
            face_roi (Image.Image): Original face image (PIL).
            target_size (tuple[int, int]): Desired image size (width, height).

        Returns:
            np.ndarray: Normalized face image array suitable for inference.
        
        Raises:
            Exception: If an error occurs during preprocessing.
        """
        try:
            face_img = face_roi.copy()
            target_w, target_h = target_size
            orig_w, orig_h = face_img.size

            scale = max(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            face_img = face_img.resize((new_w, new_h), Image.ANTIALIAS)

            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            face_img = face_img.crop((left, top, right, bottom))

            self.last_cropped_face = face_img.copy()

            face_array = np.array(face_img).astype('float32') / 255.0
            face_array = np.expand_dims(face_array, axis=0)
            return face_array

        except Exception:
            logger.error("Error in preprocess_face", exc_info=True)
            raise

    def detect_faces(self, image_path: str) -> tuple[list[tuple[int, int, int, int]], Image.Image]:
        """
        Detects faces in the given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            tuple[list[tuple[int, int, int, int]], Image.Image]:
                - A list of face bounding boxes in the format (x, y, w, h),
                - The original image (PIL.Image).
        
        Raises:
            Exception: If face detection fails.
        """
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
        except Exception:
            logger.error("Error in detect_faces", exc_info=True)
            raise

    def analyze_image(self, image_path: str) -> list[dict[str, str | tuple[int, int, int, int]]]:
        """
        Analyzes the image and predicts age, gender, and emotion
        for all detected faces.

        Args:
            image_path (str): Path to the image file.

        Returns:
            list[dict]: A list of dictionaries with results for each detected face:
                {
                    "age": str,
                    "gender": str,
                    "emotion": str,
                    "face_box": (x, y, w, h)
                }
            or error messages if analysis fails.
        """
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
                emotion_input = self.preprocess_face(face_roi.convert('L'), target_size=(48, 48))
                emotion_input = np.expand_dims(emotion_input, axis=-1)

                age = self.predict_feature(age_input, self.age_interpreter, 
                                       feature_classes = ["4 - 6 years old",
                                                          "7 - 8 years old",
                                                          "9 - 11 years old",
                                                          "12 - 19 years old",
                                                          "20 - 27 years old",
                                                          "28 - 35 years old",
                                                          "36 - 45 years old",
                                                          "46 - 60 years old",
                                                          "61 - 75 years old"])
                gender = self.predict_feature(gender_input, self.gender_interpreter,
                                             feature_classes = ["Female", "Male"])
                emotion = self.predict_feature(emotion_input, self.emotion_interpreter,
                                               feature_classes = ["Neutral", 
                                                                  "Happy", 
                                                                  "Surprise", 
                                                                  "Sad", 
                                                                  "Angry", 
                                                                  "Disgust", 
                                                                  "Fear"])

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

    def predict_feature(self, face_data: np.ndarray, feature_interpreter: tflite.Interpreter,
                        feature_classes: list[str]) -> str:
        """
        Runs inference on a feature (age/gender/emotion) using a TFLite model.

        Args:
            face_data (np.ndarray): Preprocessed face image data.
            feature_interpreter (tflite.Interpreter): TFLite interpreter for the feature.
            feature_classes (list[str]): List of possible feature classes.

        Returns:
            str: The predicted class label.
        
        Raises:
            Exception: If inference fails.
        """
        try:
            input_details = feature_interpreter.get_input_details()
            output_details = feature_interpreter.get_output_details()

            feature_interpreter.set_tensor(input_details[0]['index'], face_data.astype(np.float32))
            feature_interpreter.invoke()
            output_data = feature_interpreter.get_tensor(output_details[0]['index'])

            feature_idx = np.argmax(output_data)
            return feature_classes[feature_idx]
        except Exception:
            logger.error("Error in predicting feature ", exc_info=True)
            raise

    def return_cropped_face(self) -> Image.Image:
        """
        Returns the last cropped face after preprocessing.

        Returns:
            Image.Image | None: The cropped PIL face image, or None if no face was processed.
        """
        return self.last_cropped_face