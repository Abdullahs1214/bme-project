import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT = "Outputs/predictions(n).tsv"
OUTDIR = "Plots"


def ensure_outdir(path):
  if not os.path.exists(path):
    os.makedirs(path)


def load_data():
  df = pd.read_csv(INPUT, sep="\t")
  # Make sure label is 0/1 int
  df["label"] = df["label"].astype(int)
  return df


def plot_u2_align_by_label(df):
  """Motif / alignment module: U2 alignment score vs label."""
  fig, ax = plt.subplots(figsize=(6, 4))

  pos = df.loc[df["label"] == 1, "u2_align_score"].dropna()
  neg = df.loc[df["label"] == 0, "u2_align_score"].dropna()

  # Overlaid histograms (density) for positives vs negatives
  ax.hist(
    neg,
    bins=50,
    density=True,
    histtype="step",
    label="Non-branchpoint (label=0)",
  )
  ax.hist(
    pos,
    bins=50,
    density=True,
    histtype="step",
    label="Branchpoint (label=1)",
  )

  ax.set_xlabel("U2 alignment score")
  ax.set_ylabel("Density")
  ax.set_title("Motif / alignment signal at candidate branchpoints")
  ax.legend()

  out_path = os.path.join(OUTDIR, "feature_u2_align_by_label.png")
  fig.tight_layout()
  fig.savefig(out_path, dpi=300)
  plt.close(fig)
  print(f"Saved {out_path}")


def plot_hmm_logp_by_label(df):
  """HMM module: HMM log-probability vs label."""
  fig, ax = plt.subplots(figsize=(6, 4))

  pos = df.loc[df["label"] == 1, "hmm_logp_ctx"].dropna()
  neg = df.loc[df["label"] == 0, "hmm_logp_ctx"].dropna()

  ax.hist(
    neg,
    bins=50,
    density=True,
    histtype="step",
    label="Non-branchpoint (label=0)",
  )
  ax.hist(
    pos,
    bins=50,
    density=True,
    histtype="step",
    label="Branchpoint (label=1)",
  )

  ax.set_xlabel("HMM log-probability of context")
  ax.set_ylabel("Density")
  ax.set_title("HMM sequence-context score at candidate branchpoints")
  ax.legend()

  out_path = os.path.join(OUTDIR, "feature_hmm_logp_by_label.png")
  fig.tight_layout()
  fig.savefig(out_path, dpi=300)
  plt.close(fig)
  print(f"Saved {out_path}")


def plot_nuss_features_scatter(df, max_neg=5000):
  """Nussinov / RNA folding module: energy vs unpaired status."""
  fig, ax = plt.subplots(figsize=(6, 4))

  pos = df.loc[df["label"] == 1, ["nuss_min_energy", "nuss_unpaired"]].dropna()
  neg = df.loc[df["label"] == 0, ["nuss_min_energy", "nuss_unpaired"]].dropna()

  # Subsample negatives so the plot isn't a huge blob
  if len(neg) > max_neg:
    neg = neg.sample(n=max_neg, random_state=0)

  ax.scatter(
    neg["nuss_min_energy"],
    neg["nuss_unpaired"],
    s=5,
    alpha=0.3,
    label="Non-branchpoint (label=0)",
  )
  ax.scatter(
    pos["nuss_min_energy"],
    pos["nuss_unpaired"],
    s=8,
    alpha=0.6,
    label="Branchpoint (label=1)",
  )

  ax.set_xlabel("Nussinov minimum window energy")
  ax.set_ylabel("Unpaired indicator (0/1)")
  ax.set_title("Structure-inspired features around candidate branchpoints")
  ax.legend()

  out_path = os.path.join(OUTDIR, "feature_nuss_energy_vs_unpaired.png")
  fig.tight_layout()
  fig.savefig(out_path, dpi=300)
  plt.close(fig)
  print(f"Saved {out_path}")


def main():
  ensure_outdir(OUTDIR)
  df = load_data()
  print("Columns:", df.columns.tolist())
  print("Positive labels:", (df["label"] == 1).sum())
  print("Negative labels:", (df["label"] == 0).sum())

  plot_u2_align_by_label(df)
  plot_hmm_logp_by_label(df)
  plot_nuss_features_scatter(df)


if __name__ == "__main__":
  main()
