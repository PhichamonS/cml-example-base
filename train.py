from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import json

# -------------------------
# Parse CLI arguments
# -------------------------
parser = argparse.ArgumentParser(description="Train a simple classifier and save metrics/plot.")
parser.add_argument("--depth", type=int, default=2, help="RandomForest max_depth (default: 2)")
args = parser.parse_args()
depth = args.depth
if depth < 1:
    raise ValueError("--depth must be >= 1")

# -------------------------
# Load data with validation
# -------------------------
data_dir = Path("data")
train_features_path = data_dir / "train_features.csv"
train_labels_path = data_dir / "train_labels.csv"
test_features_path = data_dir / "test_features.csv"
test_labels_path = data_dir / "test_labels.csv"
for p in [train_features_path, train_labels_path, test_features_path, test_labels_path]:
    if not p.is_file():
        raise FileNotFoundError(f"Missing required input file: {p}")

X_train = np.genfromtxt(str(train_features_path))
y_train = np.genfromtxt(str(train_labels_path)).ravel().astype(int)
X_test = np.genfromtxt(str(test_features_path))
y_test = np.genfromtxt(str(test_labels_path)).ravel().astype(int)

# Fit a model
random_state = 30
clf = RandomForestClassifier(max_depth=depth, random_state=random_state)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(acc)
with open("metrics.txt", "w", encoding="utf-8") as outfile:
    outfile.write("Depth: " + str(depth) + "\n")
    outfile.write("Accuracy: " + str(acc) + "\n")

with open("metrics.json", "w", encoding="utf-8") as outfile:
    json.dump(
        {"accuracy": float(acc), "depth": int(depth), "random_state": int(random_state)},
        outfile,
        indent=2,
    )
    outfile.write("\n")

# Plot it
disp = ConfusionMatrixDisplay.from_estimator(
    clf, X_test, y_test, normalize="true", cmap=plt.cm.Blues
)
plt.savefig("plot.png")
plt.close()
