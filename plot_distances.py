import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_bp_distance(pred_path: str, species: str, out_png: str) -> None:
  # Read compressed or plain TSV
  if pred_path.endswith(".gz"):
    df = pd.read_csv(pred_path, sep="\t", compression="gzip")
  else:
    df = pd.read_csv(pred_path, sep="\t")

  # Sanity check
  if "dist_to_3ss" not in df.columns:
    raise ValueError("dist_to_3ss column not found in file")

  # Use ALL rows (no label filter – these are already best-per-intron predictions)
  dist = df["dist_to_3ss"].astype(int)

  # If distances are positive (downstream), flip sign so x-axis is upstream
  if (dist > 0).mean() > 0.5:
    dist = -dist

  plt.figure(figsize=(6, 4))
  plt.hist(dist, bins=range(-80, 1, 2), edgecolor="black")

  plt.xlabel("Distance upstream of 3'SS (nt)")
  plt.ylabel("Count")
  plt.title(f"Predicted Branch Point Distance to 3'SS ({species})")

  plt.tight_layout()
  plt.savefig(out_png, dpi=300)
  plt.close()

def main():
  p = argparse.ArgumentParser()
  p.add_argument("--pred", required=True, help="predictions_best_per_intron TSV[.gz]")
  p.add_argument("--species", required=True, help="label for plot title")
  p.add_argument("--out", required=True, help="output PNG path")
  args = p.parse_args()

  plot_bp_distance(args.pred, args.species, args.out)

if __name__ == "__main__":
  main()
