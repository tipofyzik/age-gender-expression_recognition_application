# Edge-Based Age, Gender, and Expression Recognition Application

An Android application designed to perform real-time face detection and multi-attribute recognition (age, gender, and facial expression) directly on edge devices (smartphones/tablets). Built using Python, Kivy, TensorFlow Lite, and Android Java API wrappers via PyJnius.

---

## Project Overview

Visual recognition of age, gender, and facial expressions on edge devices enables fast, low-latency, and privacy-preserving inferencing without relying on cloud servers. Potential use cases include targeted product recommendations in retail environments, automated patient routing in healthcare facilities, and local video analytics.

### Key Features
- **On-Device Inference:** Runs lightweight `.tflite` models locally on mobile hardware without requiring cloud APIs or internet connection.
- **Dual Input Modes:** Supports capturing real-time photos using the front camera or picking existing images from the device gallery.
- **Cross-Platform Compatibility:** Dynamic runtime backend loader (`tflite_runtime` on Android ARM devices and `tensorflow.lite` on desktop/PC) for seamless testing, development, and evaluation across platforms.
- **Complete Evaluation Suite:** Benchmark scripts included to measure model accuracy, precision, recall, F1-scores, latency, and throughput on both PC and mobile hardware.

---

## Architecture & OOP Design

The application follows Object-Oriented Programming (OOP) principles to strictly separate user interface components, native Android integration, and machine learning inference pipelines.

```
+-------------------------------------------------------------------+
|                        MobileCamera (Kivy UI)                     |
|  - Manages UI Layout & Camera Preview                             |
|  - Handles Permission Requests (Camera, Storage)                  |
|  - Performs Image Rotation Transforms (+90°)                      |
+---------------------------------+---------------------------------+
                                  |
         +------------------------+------------------------+
         |                                                 |
         v                                                 v
+-------------------------------+               +-------------------------------+
|         GalleryPicker         |               |      AttributesPredictor      |
| - Invokes Android Intent      |               | - Face Detection (fdlite)     |
| - Resolves File Paths via     |               | - Face Preprocessing & Crop   |
|   Java MediaStore API         |               | - TFLite Inference Engines:   |
+-------------------------------+               |   * MobileNet_Age.tflite      |
                                                |   * MobileNet_Gender.tflite   |
                                                |   * emotion_model.tflite      |
                                                +-------------------------------+
```

### Module Breakdown
* `main.py` — Entry point that launches the Kivy application loop.
* `MobileCamera.py` — Manages UI layouts, dynamic camera rotation (+90° orientation fix for portrait mode), permission callbacks, and user interaction.
* `GalleryPicker.py` — Interfaces with native Android Java APIs (`android.content.Intent`, `MediaStore`) via PyJnius to select images from the user's gallery.
* `AttributesPredictor.py` — Handles face detection using MediaPipe (`fdlite`), ROI cropping, normalization, and TFLite model inference for age, gender, and emotion recognition.
* `buildozer.spec` — Buildozer configuration specifying Android build requirements, API levels, permissions, and dependencies.
* `PerformanceEvaluator.py` & `main_eval.py` — Evaluation scripts for batch processing dataset benchmarks on PC or mobile hardware.
* `TransformPhoneResultsToMetrics.py` — Converts raw phone evaluation log outputs into standard classification metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix).

---

## Pretrained Models & Sources

1. **Face Detection:** Adapted from [face-detection-tflite](https://github.com/patlevin/face-detection-tflite) with custom fixes to support `tflite_runtime` directly on mobile ARM architectures.
2. **Age & Gender Classification:** Pretrained MobileNet v1 models converted to `.tflite` format. Source: [radualexandrub/Age-Gender-Classification-on-RaspberryPi4-with-TFLite-PyQt5](https://github.com/radualexandrub/Age-Gender-Classification-on-RaspberryPi4-with-TFLite-PyQt5).
3. **Facial Expression Recognition (FER):** Lightweight emotion model converted to `.tflite`. Source: [vicksam/fer-model](https://github.com/vicksam/fer-model).

---

## Performance & Benchmark Evaluation

### 1. Classification Performance Metrics

#### Gender Recognition Model
* **Dataset:** Gender Recognition Dataset (Kaggle) — Test Subfolder

| Class | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Female** | **0.8763** | 0.9275 | 0.7420 | 0.8244 |
| **Male** | **0.8529** | 0.9627 | 0.9045 | — |

#### Age Recognition Model
* **Overall Model Accuracy:** **0.4773**
* **Dataset:** Facial Age Dataset (Kaggle)

| Age Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **4 - 6 years old** | 0.6348 | 0.8233 | 0.7169 |
| **7 - 8 years old** | 0.0000 | 0.0000 | 0.0000 |
| **9 - 11 years old** | 0.0497 | 0.0841 | — |
| **12 - 19 years old** | 0.5656 | 0.5116 | — |
| **20 - 27 years old** | 0.4881 | 0.4847 | — |
| **28 - 35 years old** | 0.3911 | 0.3534 | — |
| **36 - 45 years old** | 0.2783 | 0.3054 | — |
| **46 - 60 years old** | 0.7045 | 0.5837 | — |
| **61 - 75 years old** | 0.4302 | 0.5526 | — |

#### Facial Emotion Recognition Model
* **Overall Model Accuracy:** **0.6172**
* **Dataset:** Face Expression Recognition Dataset (Kaggle) — Validation Subfolder

| Emotion Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Neutral** | 0.4115 | 0.8798 | 0.5607 |
| **Happy** | 0.8815 | 0.8855 | 0.8835 |
| **Surprise** | 0.7684 | 0.7425 | — |
| **Sad** | 0.3513 | 0.4716 | — |
| **Angry** | 0.4471 | 0.5362 | — |
| **Disgust** | 0.2062 | 0.3252 | — |
| **Fear** | 0.1522 | 0.2520 | — |
| **Contempt** | 0.0000 | 0.0000 | 0.0000 |

---

### 2. Device Hardware Performance
* **Benchmark Dataset:** 1279 RGB images
* **Mobile Hardware:** Octa-core CPU (2x2.3 GHz Kryo 470 Gold & 6x1.8 GHz Kryo 470 Silver)
* **PC Hardware:** AMD Ryzen 5 5500U @ 2.10 GHz

| Execution Target / Mode | Total Time | Throughput Rate |
| :--- | :---: | :---: |
| **PC CPU (AMD Ryzen 5)** | 47 sec | **27.21 img/s** |
| **Mobile Phone (Foreground)** | 2 min, 46 sec | **7.70 img/s** |
| **Mobile Phone (Background)** | 12 min, 48 sec | **1.67 img/s** |

---

## Installation & Build Instructions

### Prerequisites
* Python 3.8+
* `kivy == 2.3.0`
* `numpy == 1.22.3`
* `pillow == 8.4.0`
* `tflite-runtime == 2.8.0` (for mobile ARM platforms)
* `tensorflow` (for desktop/PC development and evaluation)

---

### Compiling APK with Buildozer (Ubuntu Linux)

Building an `.apk` package requires a Linux environment (Ubuntu is recommended).

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/tipofyzik/age-gender-expression_recognition_application.git
   cd age-gender-expression_recognition_application
   ```

2. **Verify Directory Structure:**
   Ensure the required files and folders are present in the project root:
   ```text
   ├── AttributesPredictor.py
   ├── GalleryPicker.py
   ├── MobileCamera.py
   ├── main.py
   ├── buildozer.spec
   ├── libs/
   │   └── fdlite/
   └── models/
       ├── MobileNet_Age.tflite
       ├── MobileNet_Gender.tflite
       └── emotion_model.tflite
   ```

3. **Install Build Dependencies:**
   ```bash
   sudo apt update
   sudo apt install -y git zip unzip openjdk-17-jdk python3-pip autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libsqlite3-dev cmake libssl-dev
   pip install --user buildozer
   ```

4. **Build the APK:**
   ```bash
   buildozer -v android debug
   ```
   The generated `.apk` file will be located in the `bin/` directory.

---

## Evaluation Instructions

To reproduce evaluation results:

1. **PC Evaluation:**
   Execute the evaluation pipeline on PC:
   ```bash
   python PerformanceEvaluator.py
   ```
   Results will be saved in `./evaluation_results/pc/`.

2. **Mobile Device Evaluation:**
   - Swap `main.py` with `main_eval.py`.
   - Transfer target datasets to your device's `Downloads` folder.
   - Build and run the app; evaluation results will be saved to `Downloads/results/`.
   - Run `TransformPhoneResultsToMetrics.py` to calculate final metrics from device CSV output files.

---

## Repository Layout

```text
.
├── AttributesPredictor.py            # TFLite inference engine & image processing
├── GalleryPicker.py                  # Android Gallery Intent wrapper (PyJnius)
├── MobileCamera.py                   # Main Kivy UI & camera capture workflow
├── PerformanceEvaluator.py           # PC batch model evaluation script
├── TransformPhoneResultsToMetrics.py # Tool to parse device log CSVs into metrics
├── buildozer.spec                    # Buildozer Android configuration file
├── libs/                             # Modified fdlite library for TFLite runtime
├── main.py                           # App entry point
├── main_eval.py                      # On-device batch evaluation entry point
└── models/                           # Pretrained .tflite weights
```

---

## References & Citation

* [patlevin/face-detection-tflite](https://github.com/patlevin/face-detection-tflite) for face detection implementation.
* [radualexandrub](https://github.com/radualexandrub/Age-Gender-Classification-on-RaspberryPi4-with-TFLite-PyQt5) for MobileNet age and gender classification models.
* [vicksam](https://github.com/vicksam/fer-model) for FER emotion classification models.
* Datasets provided by Kaggle users Rashik Rahman, Fazle Rabbi, and Jonathan Oheix.


