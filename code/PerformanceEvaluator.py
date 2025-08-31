# Data preparation for performance evaluation.
import os
import csv
import shutil

import time


def split_photos_by_age_folders(src_root: str, dst_root: str, age_ranges: dict[str, tuple[int, int]]):
    """
    Splits potost to age ranges folders.
    
    Args:
        src_root (str): The path to the initial folder, where name of every subfolder is the age of people on photos.
        dst_root (str): Path to a final folder, where folders with age ranges will be created.
        age_ranges (dict): Dictionary with age ranges {"4-6": (4,6), "7-8": (7,8), ...}
    """
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # Go through all folders with age name
    for age_folder in os.listdir(src_root):
        age_folder_path = os.path.join(src_root, age_folder)
        if not os.path.isdir(age_folder_path):
            continue
        
        try:
            age = int(age_folder)
        except ValueError:
            continue  # If folder name is not a number, then skip

        # Defining age range
        for folder_name, (low, high) in age_ranges.items():
            if low <= age <= high:
                target_dir = os.path.join(dst_root, folder_name)
                os.makedirs(target_dir, exist_ok=True)

                # Copying all files from the initial folder
                for fname in os.listdir(age_folder_path):
                    src_path = os.path.join(age_folder_path, fname)
                    dst_path = os.path.join(target_dir, fname)
                    shutil.copy2(src_path, dst_path)  # можно заменить на move
                break

age_ranges = {
    "0": (4, 6),
    "1": (7, 8),
    "2": (9, 11),
    "3": (12, 19),
    "4": (20, 27),
    "5": (28, 35),
    "6": (36, 45),
    "7": (46, 60),
    "8": (61, 75)
}

src_folder = "./test_datasets/age_data"
age_test_folder = "./test_datasets/age_data/sorted_photos"
emotion_test_folder = "./test_datasets/emotion_data/images"
gender_test_folder = "./test_datasets/gender_data/Test"

# split_photos_by_age_folders(src_folder, age_test_folder, age_ranges)







from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm
from AttributesPredictor import AttributesPredictor
import os
import csv

class PerformanceEvaluator():
    def __init__(self):
        pass

    def evaluate_model_on_dataset(self, predictor: AttributesPredictor, attribute: str, dataset_path: str):
        """
        Оценивает модель (age, gender или emotion) на тестовом датасете.
        Использует тот же пайплайн обработки изображений, что и AttributesPredictor.
        """
        y_true = []
        y_pred = []

        if attribute == "age":
            classes = ["4 - 6 years old",
                    "7 - 8 years old",
                    "9 - 11 years old",
                    "12 - 19 years old",
                    "20 - 27 years old",
                    "28 - 35 years old",
                    "36 - 45 years old",
                    "46 - 60 years old",
                    "61 - 75 years old"]
        elif attribute == "gender":
            classes = ["Female", "Male"]
        elif attribute == "emotion":
            classes = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
        else:
            raise ValueError(f"Unknown attribute: {attribute}")

        for class_folder in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_folder)
            if not os.path.isdir(class_path):
                continue

            true_class_idx = int(class_folder)
            true_class_name = classes[true_class_idx]

            for file in tqdm(os.listdir(class_path), desc=f"Processing {attribute}/{class_folder}"):
                img_path = os.path.join(class_path, file)
                try:
                    results = predictor.analyze_image(img_path)
                    if "error" in results[0]:
                        continue

                    prediction = results[0][attribute]
                    y_true.append(true_class_name)
                    y_pred.append(prediction)

                except Exception as e:
                    print(f"Ошибка при обработке {img_path}: {e}")
                    continue

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        precision = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

        return acc, cm, precision, recall, f1, classes

    def save_results_to_csv(self, attribute, accuracy, cm, precision, recall, f1, classes, output_dir="./evaluation_results/pc"):
        """
        Сохраняет результаты (accuracy, confusion matrix, precision, recall, F1, classes) в CSV для конкретного атрибута
        """
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{attribute}_results.csv")

        with open(file_path, mode="w", newline="") as f:
            writer = csv.writer(f)

            # 1. Accuracy
            writer.writerow(["**Accuracy**"])
            writer.writerow([f"{accuracy}"])
            writer.writerow([])

            # 2. Confusion matrix
            writer.writerow(["**Confusion Matrix**"])
            for row in cm:
                writer.writerow(row)
            writer.writerow([])

            # 3. Precision
            writer.writerow(["**Precision per class**"])
            for cls, val in zip(classes, precision):
                writer.writerow([cls, val])
            writer.writerow([])

            # 4. Recall
            writer.writerow(["**Recall per class**"])
            for cls, val in zip(classes, recall):
                writer.writerow([cls, val])
            writer.writerow([])

            # 5. F1-score
            writer.writerow(["**F1-score per class**"])
            for cls, val in zip(classes, f1):
                writer.writerow([cls, val])
            writer.writerow([])

            # 6. Classes
            writer.writerow(["**Classes**"])
            writer.writerow(["class name", "class index"])
            for idx, cls in enumerate(classes):
                writer.writerow([cls, idx])

        print(f"[INFO] Results saved to {file_path}")




age_test_folder = "./test_datasets/age_data/sorted_photos"
emotion_test_folder = "./test_datasets/emotion_data/images/validation"
gender_test_folder = "./test_datasets/gender_data/Test"

predictor = AttributesPredictor()
performance_evaluator = PerformanceEvaluator()

attributes = {"gender": gender_test_folder, 
              "age": age_test_folder, 
              "emotion": emotion_test_folder}

# For evaluation of a device performance
attributes = {"emotion": emotion_test_folder}

for attribute, dataset_path in attributes.items():
    start = time.perf_counter()

    acc, cm, precision, recall, f1, classes = performance_evaluator.evaluate_model_on_dataset(
        predictor,
        attribute = attribute,
        dataset_path = dataset_path
    )

    end = time.perf_counter()
    print("Time passed:", time.strftime("%H:%M:%S", time.gmtime(end - start)))
    performance_evaluator.save_results_to_csv(attribute, acc, cm, precision, recall, f1, classes)


