import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os

def evaluate_results_csv(csv_path: str, attribute: str):
    """
    Reads CSV with phone results (true_class_idx, pred_class_idx) and computes
    confusion matrix, accuracy, precision, recall, and F1-score.
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[WARN] CSV is empty: {csv_path}")
        return

    true_labels = df['true_class_idx'].tolist()
    pred_labels = df['pred_class_idx'].tolist()

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average=None, zero_division=0)
    recall = recall_score(true_labels, pred_labels, average=None, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0)

    # Save to CSV
    output_csv = csv_path.replace("_phone_results.csv", f"_{attribute}_sklearn_metrics.csv")
    with open(output_csv, "w", newline="") as f:
        f.write("**Accuracy**\n")
        f.write(f"{acc}\n\n")

        f.write("**Confusion Matrix**\n")
        cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in range(cm.shape[0])],
                             columns=[f"pred_{i}" for i in range(cm.shape[1])])
        cm_df.to_csv(f, header=True)
        f.write("\n")

        f.write("**Precision per class**\n")
        for i, val in enumerate(precision):
            f.write(f"class_{i},{val}\n")
        f.write("\n")

        f.write("**Recall per class**\n")
        for i, val in enumerate(recall):
            f.write(f"class_{i},{val}\n")
        f.write("\n")

        f.write("**F1-score per class**\n")
        for i, val in enumerate(f1):
            f.write(f"class_{i},{val}\n")
        f.write("\n")

    print(f"Metrics saved to {output_csv}")


csv_folder = "./evaluation_results/phone"
attributes = ["age", "gender", "emotion"]

for attribute in attributes:
    csv_path = os.path.join(csv_folder, f"{attribute}_phone_results.csv")
    evaluate_results_csv(csv_path, attribute)
    