import os
import csv
from AttributesPredictor import AttributesPredictor


def get_classes(attribute: str):
    """Возвращает список классов по атрибуту."""
    if attribute == "age":
        return [
            "4 - 6 years old",
            "7 - 8 years old",
            "9 - 11 years old",
            "12 - 19 years old",
            "20 - 27 years old",
            "28 - 35 years old",
            "36 - 45 years old",
            "46 - 60 years old",
            "61 - 75 years old",
        ]
    if attribute == "gender":
        return ["Female", "Male"]
    if attribute == "emotion":
        return ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
    raise ValueError(f"Unknown attribute: {attribute}")

def collect_dataset_images(dataset_path: str):
    """
    Возвращает список кортежей (image_path, true_class_idx).
    Подпапки должны быть пронумерованы (0, 1, 2, ...).
    Поддерживаются только файлы с расширениями .jpg, .jpeg, .png
    """
    dataset = []

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path does not exist: {dataset_path}")
        return dataset

    class_folders = os.listdir(dataset_path)
    if not class_folders:
        print(f"[DEBUG] No subfolders found in {dataset_path}")
        return dataset

    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        try:
            true_class_idx = int(class_folder)
        except ValueError:
            print(f"[WARN] Skipping folder '{class_folder}' (not a number)")
            continue

        files = os.listdir(class_path)
        img_count = 0
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                dataset.append((os.path.join(class_path, fname), true_class_idx))
                img_count += 1

        print(f"[DEBUG] Found {img_count} images in class '{class_folder}'")

    print(f"[DEBUG] Total images collected: {len(dataset)}")
    return dataset


def process_image(predictor: AttributesPredictor, img_path: str, attribute: str):
    """Прогоняет одно изображение через модель."""
    try:
        results = predictor.analyze_image(img_path)
        if "error" in results[0]:
            return None
        return results[0][attribute]
    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")
        return None


def write_results(file_path: str, rows: list):
    """Сохраняет результаты в CSV."""
    with open(file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_class_idx", "pred_class_idx", "pred_class_name"])
        writer.writerows(rows)


def evaluate_on_phone(predictor: AttributesPredictor, attribute: str, dataset_path: str, output_dir: str) -> str:
    """
    Прогоняет датасет на телефоне и сохраняет CSV с результатами (без sklearn).
    Каждая строка = [image_path, true_class_idx, predicted_class_idx, predicted_class_name].
    """
    os.makedirs(output_dir, exist_ok=True)
    classes = get_classes(attribute)
    dataset = collect_dataset_images(dataset_path)
    print(f"[DEBUG] Found {len(dataset)} images in {dataset_path}")

    rows = []
    total = len(dataset)
    for i, (img_path, true_idx) in enumerate(dataset, start=1):
        pred_class = process_image(predictor, img_path, attribute)
        if pred_class is None:
            continue
        try:
            pred_idx = classes.index(pred_class)
        except ValueError:
            print(f"[WARN] Unexpected class '{pred_class}' for {img_path}")
            continue
        rows.append([img_path, true_idx, pred_idx, pred_class])

        # Простейший прогресс-бар
        if i % 10 == 0 or i == total:
            print(f"[INFO] Processed {i}/{total} images for {attribute}")

    file_path = os.path.join(output_dir, f"{attribute}_phone_results.csv")
    write_results(file_path, rows)
    print(f"[INFO] Results saved to {file_path}")
    return file_path



from jnius import autoclass
def get_downloads_path() -> str:
    """
    Retrieves the absolute path to the Android Downloads directory.

    Returns:
        str: The path to the Downloads folder.
    """
    Environment = autoclass('android.os.Environment')
    downloads_dir = Environment.getExternalStoragePublicDirectory(
        Environment.DIRECTORY_DOWNLOADS
    ).getAbsolutePath()
    return downloads_dir


import time

if __name__ == "__main__":
    predictor = AttributesPredictor()

    age_test_folder = "test_datasets/age_data/sorted_photos"
    emotion_test_folder = "test_datasets/emotion_data/images/validation"
    gender_test_folder = "test_datasets/gender_data/Test"

    attributes = {"gender": gender_test_folder, 
              "age": age_test_folder, 
              "emotion": emotion_test_folder}
    
    # For evaluation of a device performance
    attributes = {"emotion": emotion_test_folder}

    download_path = get_downloads_path()
    output_dir = os.path.join(download_path, "results")
    os.makedirs(output_dir, exist_ok=True)

    for attribute, dataset_path in attributes.items():
        dataset_path = os.path.join(download_path, dataset_path)

        start = time.perf_counter()

        evaluate_on_phone(predictor, attribute, dataset_path, output_dir)

        end = time.perf_counter()
        print("Time passed:", time.strftime("%H:%M:%S", time.gmtime(end - start)))

