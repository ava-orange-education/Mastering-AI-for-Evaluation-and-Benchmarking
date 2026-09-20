import pandas as pd from sklearn.metrics import accuracy_score,
precision_score, recall_score, f1_score

Load the evaluation dataset

data = pd.read_csv(“data/evaluation_dataset.csv”)

Get actual and predicted labels

actual = data[“actual_label”] predicted = data[“predicted_label”]

Calculate evaluation metrics

accuracy = accuracy_score(actual, predicted) precision =
precision_score(actual, predicted, average=“weighted”) recall =
recall_score(actual, predicted, average=“weighted”) f1 =
f1_score(actual, predicted, average=“weighted”)

Display evaluation results

print(“Accuracy:”, accuracy) print(“Precision:”, precision)
print(“Recall:”, recall) print(“F1 Score:”, f1)
