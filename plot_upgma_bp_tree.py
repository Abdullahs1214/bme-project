import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram


def load_distances(path, min_d=-60, max_d=-10):
  """
  Read predictions_best_per_intron.tsv or .tsv.gz and return
  the dist_to_3ss values in the canonical BP window.
  """
  if not os.path.exists(path):
    raise FileNotFoundError(f"File not found: {path}")

  df = pd.read_csv(path, sep="\t", compression="infer")
  if "dist_to_3ss" not in df.columns:
    raise ValueError(f"'dist_to_3ss' column not found in {path}")

  d = df["dist_to_3ss"].astype(float)
  d = d[(d >= min_d) & (d <= max_d)]
  return d.to_numpy()


def make_hist(distances, bins):
  """
  Turn a 1D array of distances into a normalized histogram vector.
  """
  hist, _ = np.histogram(distances, bins=bins)
  hist = hist.astype(float)
  if hist.sum() > 0:
    hist /= hist.sum()
  return hist


def build_distance_matrix(hists, names):
  """
  Compute pairwise L2 distances between histogram vectors.
  Returns an (n x n) symmetric matrix.
  """
  n = len(names)
  D = np.zeros((n, n), dtype=float)
  for i in range(n):
    for j in range(i + 1, n):
      d = np.linalg.norm(hists[i] - hists[j])
      D[i, j] = d
      D[j, i] = d
  return D


def plot_upgma_tree(D, names, out_png):
  """
  Use average-linkage hierarchical clustering (UPGMA) to build a tree
  and plot it as a dendrogram.
  """
  # SciPy wants a condensed distance vector
  condensed = squareform(D)
  Z = linkage(condensed, method="average")  # UPGMA

  plt.figure(figsize=(6, 4))
  dendrogram(Z, labels=names, orientation="right")
  plt.xlabel("L2 distance between BP distance distributions")
  plt.title("UPGMA tree from predicted BP distances")
  plt.tight_layout()
  plt.savefig(out_png, dpi=300)
  plt.close()
  print(f"Saved tree figure to {out_png}")


def main():
  parser = argparse.ArgumentParser(
    description="Build UPGMA tree from BP distance distributions."
  )
  parser.add_argument("--human", required=True,
                      help="Human predictions_best_per_intron.tsv(.gz)")
  parser.add_argument("--chimp", required=True,
                      help="Chimpanzee predictions_best_per_intron.tsv(.gz)")
  parser.add_argument("--rhesus", required=True,
                      help="Rhesus macaque predictions_best_per_intron.tsv(.gz)")
  parser.add_argument("--mouse", required=True,
                      help="Mouse predictions_best_per_intron.tsv(.gz)")
  parser.add_argument("--rat", required=True,
                      help="Rat predictions_best_per_intron.tsv(.gz)")
  parser.add_argument("--out", required=True,
                      help="Output PNG path for the UPGMA tree figure")
  parser.add_argument("--min_dist", type=float, default=-60,
                      help="Minimum distance to 3'SS to include (default -60)")
  parser.add_argument("--max_dist", type=float, default=-10,
                      help="Maximum distance to 3'SS to include (default -10)")
  args = parser.parse_args()

  species = [
    ("Human", args.human),
    ("Chimpanzee", args.chimp),
    ("Rhesus macaque", args.rhesus),
    ("Mouse", args.mouse),
    ("Rat", args.rat),
  ]

  # Common histogram bins across all species
  bins = np.arange(args.min_dist - 0.5, args.max_dist + 1.5, 1.0)

  names = []
  hists = []

  for name, path in species:
    print(f"Loading distances for {name} from {path} ...")
    d = load_distances(path, args.min_dist, args.max_dist)
    print(f"  {len(d)} sites in window [{args.min_dist}, {args.max_dist}]")
    h = make_hist(d, bins)
    names.append(name)
    hists.append(h)

  D = build_distance_matrix(hists, names)
  print("Pairwise L2 distance matrix:")
  print(pd.DataFrame(D, index=names, columns=names))

  plot_upgma_tree(D, names, args.out)


if __name__ == "__main__":
  main()
