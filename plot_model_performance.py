import json
import os

import matplotlib.pyplot as plt
import numpy as np

def load_metrics(path):
  with open(path) as f:
    return json.load(f)

def plot_model_performance(metrics, out_png):
  # Models to display (must match keys in train_metrics.json)
  models = ["RandomForestClassifier", "LogisticRegression", "GaussianNB"]

  acc = [metrics[m]["Accuracy"] for m in models]
  f1 = [metrics[m]["F1 Score"] for m in models]
  auc = [metrics[m]["ROC AUC"] for m in models]

  x = np.arange(len(models))
  width = 0.25

  fig, ax = plt.subplots(figsize=(8, 4))

  ax.bar(x - width, acc, width, label="Accuracy")
  ax.bar(x, f1, width, label="F1 Score")
  ax.bar(x + width, auc, width, label="ROC AUC")

  ax.set_xticks(x)
  ax.set_xticklabels(["Random Forest", "Logistic Reg.", "Gaussian NB"])
  ax.set_ylabel("Score")
  ax.set_ylim(0, 1.05)
  ax.set_title("Lazy-BP: Model Performance on Human chr12")
  ax.legend(loc="best")
  ax.grid(axis="y", linestyle="--", alpha=0.4)

  fig.tight_layout()
  fig.savefig(out_png, dpi=300)
  plt.close(fig)

def main():
  metrics_path = os.path.join("Outputs_lazy", "train_metrics.json")
  out_png = os.path.join("Outputs_lazy", "fig_model_performance.png")

  metrics = load_metrics(metrics_path)
  plot_model_performance(metrics, out_png)
  print(f"Saved figure to {out_png}")

if __name__ == "__main__":
  main()
