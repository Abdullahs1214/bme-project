"""
Branchpoint prediction pipeline (Python)

What it does
  1) Loads introns and a reference genome.
  2) Generates candidate branchpoint sites (A's) 10-60 nt upstream of the 3' splice site.
  3) Engineers biologically-informed features:
       - local sequence k-mer context (±5 nt window)
       - distance-to-3'ss
       - pyrimidine-tract (polyY) statistics in the 3' intron region
       - simple U2-duplex heuristic score against a U2-consistent motif
       - alignment score to a U2-like motif (Module 2)
       - HMM branchpoint features (Module 4)
       - Nussinov-based structure features (Module 5)
  4) Trains/evaluates models (logistic regression, random forest, MLP) if labels are supplied.
  5) Scores candidates genome-wide and writes ranked predictions.
"""

import argparse, os, sys, json, gzip
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from joblib import dump
from pyfaidx import Fasta as FaidxFasta

import math

DNA = set("ACGTN")
KMER_K = 3   # small and robust for sparse data
SEQ_WIN = 5  # ±5 around candidate A -> 11-mer

# ---------- HMM model (Module 4) ----------

HMM_STATES = ['N', 'B']

HMM_START_PROBS = {
  'N': 0.9,
  'B': 0.1,
}

HMM_TRANS_PROBS = {
  'N': {'N': 0.9, 'B': 0.1},
  'B': {'N': 0.3, 'B': 0.7},
}

HMM_EMIT_PROBS = {
  'N': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
  'B': {'A': 0.6,  'C': 0.15, 'G': 0.1,  'T': 0.15},
}

# ---------- Nussinov-based minimum-energy folding (Module 5) ----------

def nuss_energy(a, b):
  pair = (a, b)
  if pair in [('A', 'U'), ('U', 'A'), ('G', 'U'), ('U', 'G')]:
    return -2.0
  if pair in [('G', 'C'), ('C', 'G')]:
    return -3.0
  return None

def nussinov_min_energy_window(seq):
  """
  Modified Nussinov:
    - no loops < 3
    - internal unpaired bases: +0.1
    - external unpaired bases: 0
  Returns (min_energy, paired_list) where paired_list[i] = j or -1.
  """
  seq = seq.upper()
  n = len(seq)
  if n == 0:
    return 0.0, []

  M = [[0.0] * n for _ in range(n)]
  trace = [[None] * n for _ in range(n)]

  for length in range(1, n):
    for i in range(n - length):
      j = i + length

      # i unpaired
      opt_i_unpaired = M[i + 1][j] + (0.1 if i != 0 else 0.0)
      best = opt_i_unpaired
      trace[i][j] = ('i_unpaired', i + 1, j)

      # j unpaired
      opt_j_unpaired = M[i][j - 1] + (0.1 if j != n - 1 else 0.0)
      if opt_j_unpaired < best:
        best = opt_j_unpaired
        trace[i][j] = ('j_unpaired', i, j - 1)

      # i–j pair
      e = nuss_energy(seq[i], seq[j])
      if e is not None and (j - i - 1) >= 3:
        opt_pair = M[i + 1][j - 1] + e
        if opt_pair < best:
          best = opt_pair
          trace[i][j] = ('pair', i + 1, j - 1)

      # bifurcation
      for k in range(i, j):
        opt_split = M[i][k] + M[k + 1][j]
        if opt_split < best:
          best = opt_split
          trace[i][j] = ('split', i, k, k + 1, j)

      M[i][j] = best

  paired = [-1] * n

  def tb(i, j):
    if i >= j:
      return
    action = trace[i][j]
    if action is None:
      return
    kind = action[0]
    if kind == 'i_unpaired':
      _, ni, nj = action
      tb(ni, nj)
    elif kind == 'j_unpaired':
      _, ni, nj = action
      tb(ni, nj)
    elif kind == 'pair':
      _, ni, nj = action
      paired[i] = j
      paired[j] = i
      tb(ni, nj)
    elif kind == 'split':
      _, i1, k, k2, j2 = action
      tb(i1, k)
      tb(k2, j2)

  tb(0, n - 1)
  return M[0][n - 1], paired

def nuss_features(ctx, center_index):
  """
  ctx: sequence window around candidate A
  center_index: index of candidate A in ctx
  Returns (min_energy, is_unpaired_at_center)
  """
  if not ctx:
    return 0.0, 1
  e, paired = nussinov_min_energy_window(ctx)
  if not paired or center_index >= len(paired):
    return float(e), 1
  is_unpaired = 1 if paired[center_index] == -1 else 0
  return float(e), int(is_unpaired)

# ---------- HMM helpers (Module 4) ----------

def _log(x):
  if x <= 0.0:
    return -1e9
  return math.log(x)

def _logsumexp(vals):
  m = max(vals)
  if m <= -1e8:
    return -1e9
  s = sum(math.exp(v - m) for v in vals)
  return m + math.log(s)

def hmm_forward_log(seq, states, start_p, trans_p, emit_p):
  seq = seq.upper()
  T = len(seq)
  if T == 0:
    return -1e9, []

  F = []
  sym0 = seq[0]
  row0 = {}
  for s in states:
    e = emit_p[s].get(sym0, 1e-6)
    row0[s] = _log(start_p[s]) + _log(e)
  F.append(row0)

  for t in range(1, T):
    sym = seq[t]
    row = {}
    for s in states:
      e = emit_p[s].get(sym, 1e-6)
      vals = []
      for sp in states:
        vals.append(F[t - 1][sp] + _log(trans_p[sp][s]))
      row[s] = _log(e) + _logsumexp(vals)
    F.append(row)

  logP = _logsumexp([F[T - 1][s] for s in states])
  return logP, F

def hmm_backward_log(seq, states, trans_p, emit_p):
  seq = seq.upper()
  T = len(seq)
  if T == 0:
    return []

  B = []
  row_last = {s: 0.0 for s in states}
  B.append(row_last)

  for t in range(T - 2, -1, -1):
    sym_next = seq[t + 1]
    row = {}
    for s in states:
      vals = []
      for sp in states:
        e = emit_p[sp].get(sym_next, 1e-6)
        vals.append(_log(trans_p[s][sp]) + _log(e) + B[-1][sp])
      row[s] = _logsumexp(vals)
    B.append(row)

  B.reverse()
  return B

def hmm_branch_features(ctx11):
  """
  HMM features on the 11-nt context:
    - logP(sequence | HMM)
    - posterior P(state=B at center)
  """
  ctx11 = ctx11.upper()
  if len(ctx11) == 0:
    return 0.0, 0.0

  logP, F = hmm_forward_log(ctx11, HMM_STATES, HMM_START_PROBS, HMM_TRANS_PROBS, HMM_EMIT_PROBS)
  Bmat = hmm_backward_log(ctx11, HMM_STATES, HMM_TRANS_PROBS, HMM_EMIT_PROBS)

  center = len(ctx11) // 2
  num = F[center]['B'] + Bmat[center]['B']
  den = _logsumexp([F[center][s] + Bmat[center][s] for s in HMM_STATES])
  pB_center = math.exp(num - den) if den > -1e8 else 0.0

  return float(logP), float(pB_center)

# ---------- Local alignment (Module 2) ----------

def local_alignment_score(seq, motif, match=2, mismatch=-1, gap=-2):
  """
  Smith–Waterman local alignment score between seq and motif.
  Returns best local alignment score (no traceback).
  """
  seq = seq.upper()
  motif = motif.upper()
  m, n = len(seq), len(motif)
  if m == 0 or n == 0:
    return 0.0

  H = [[0] * (n + 1) for _ in range(m + 1)]
  best = 0

  for i in range(1, m + 1):
    for j in range(1, n + 1):
      s = match if seq[i - 1] == motif[j - 1] else mismatch
      diag = H[i - 1][j - 1] + s
      up = H[i - 1][j] + gap
      left = H[i][j - 1] + gap
      val = max(0, diag, up, left)
      H[i][j] = val
      if val > best:
        best = val

  return float(best)

U2_MOTIF = "TACTAAC"

def u2_alignment_score(ctx11, motif=U2_MOTIF):
  ctx11 = ctx11.upper()
  if 'N' in ctx11:
    return 0.0
  return local_alignment_score(ctx11, motif)

# ---------- Core data loading / feature helpers ----------

def read_introns(path):
  df = pd.read_csv(path, sep='\t', header=0)
  needed = {'chrom', 'start', 'end', 'strand', 'three_ss'}
  missing = needed - set(df.columns)
  if missing:
    raise ValueError(f"Missing columns in intron file: {missing}")
  df = df.copy()
  df['chrom'] = df['chrom'].astype(str)
  df['start'] = df['start'].astype(int)
  df['end'] = df['end'].astype(int)
  df['three_ss'] = df['three_ss'].astype(int)
  return df

def revcomp(seq):
  comp = str.maketrans("ACGTN", "TGCAN")
  return seq.translate(comp)[::-1]

class Fasta:
  def __init__(self, fasta_path):
    self.fa = FaidxFasta(fasta_path, as_raw=True)

  def fetch(self, chrom, start, end, strand='+'):
    if start < 0:
      start = 0
    if end <= start:
      return ""
    try:
      seq = str(self.fa[chrom][start:end]).upper()
    except KeyError:
      return ""
    if strand == '-':
      seq = revcomp(seq)
    return seq

def is_valid_seq(seq):
  return set(seq) <= DNA

def intron_region_for_polyY(intron_row, flank=60):
  if intron_row['strand'] == '+':
    region_start = max(intron_row['start'], intron_row['three_ss'] - flank)
    region_end = intron_row['three_ss']
  else:
    region_start = intron_row['three_ss']
    region_end = min(intron_row['end'], intron_row['three_ss'] + flank)
  return int(region_start), int(region_end)

def enumerate_candidates(intron_row, window_min=10, window_max=60):
  strand = intron_row['strand']
  three = intron_row['three_ss']
  if strand == '+':
    start = three - window_max
    end = three - window_min
    coords = list(range(start, end + 1))
  else:
    start = three + window_min
    end = three + window_max
    coords = list(range(start, end + 1))
  return coords

def u2_heuristic_score(ctx11):
  s = ctx11
  if len(s) != 11 or 'N' in s:
    return 0.0
  score = 1.0 if s[5] == 'A' else -1.0
  y = set('CT')
  if s[3] in y:
    score += 0.5
  if s[4] in y:
    score += 0.5
  if s[4] == 'G':
    score -= 0.3
  if s[6] == 'G':
    score -= 0.3
  down = s[6:11]
  score += 0.1 * sum(1 for c in down if c in y)
  up = s[0:5]
  score += 0.05 * sum(1 for c in up if c in set('AT'))
  return float(score)

def polyY_features(seq):
  if not seq:
    return 0.0, 0, 0
  y = set('CT')
  arr = [1 if c in y else 0 for c in seq]
  frac = sum(arr) / len(arr)
  longest = 0
  cur = 0
  for v in arr:
    if v == 1:
      cur += 1
      longest = max(longest, cur)
    else:
      cur = 0
  y3 = 0
  for i in range(0, len(seq) - 2):
    if seq[i] in y and seq[i + 1] in y and seq[i + 2] in y:
      y3 += 1
  return float(frac), int(longest), int(y3)

def kmers(s, k):
  return [s[i:i + k] for i in range(0, len(s) - k + 1)]

def make_kmer_vocab(seqs, k=KMER_K, top=256):
  cnt = Counter()
  for s in seqs:
    for m in kmers(s, k):
      if set(m) <= set('ACGT'):
        cnt[m] += 1
  vocab = [m for m, _ in cnt.most_common(top)]
  index = {m: i for i, m in enumerate(vocab)}
  return vocab, index

def onehot_kmers(s, index, k=KMER_K):
  vec = np.zeros(len(index), dtype=np.float32)
  for m in kmers(s, k):
    j = index.get(m)
    if j is not None:
      vec[j] += 1.0
  if vec.sum() > 0:
    vec = vec / vec.sum()
  return vec

def load_labels(path):
  lab = pd.read_csv(path, sep='\t', header=0)
  need = {'chrom', 'pos', 'strand'}
  if need - set(lab.columns):
    raise ValueError("labels file must have columns: chrom,pos,strand")
  lab['pos'] = lab['pos'].astype(int)
  lab['key'] = (
    lab['chrom'].astype(str)
    + ':' + lab['pos'].astype(str)
    + ':' + lab['strand'].astype(str)
  )
  return set(lab['key'].tolist())

# ---------- Build dataset ----------

def build_dataset(introns_df, fasta, labels=None, window_min=10, window_max=60, outdir=Path('.')):
  rows = []
  seqs_for_vocab = []
  for _, r in introns_df.iterrows():
    cand_positions = enumerate_candidates(r, window_min, window_max)
    for pos in cand_positions:
      if r['strand'] == '+':
        s = pos - SEQ_WIN
        e = pos + SEQ_WIN + 1
      else:
        s = pos - SEQ_WIN
        e = pos + SEQ_WIN + 1
      try:
        ctx = fasta.fetch(r['chrom'], s, e, strand=r['strand'])
      except Exception:
        continue
      if len(ctx) != 11 or not is_valid_seq(ctx):
        continue
      if ctx[5] != 'A':
        continue

      # polyY window near 3'ss
      pS, pE = intron_region_for_polyY(r, flank=60)
      polyY_seq = fasta.fetch(r['chrom'], pS, pE, strand=r['strand'])
      fracY, runY, y3 = polyY_features(polyY_seq)

      dist = abs(r['three_ss'] - pos)
      u2s = u2_heuristic_score(ctx)
      u2_align = u2_alignment_score(ctx)
      hmm_logp, hmm_pB = hmm_branch_features(ctx)

      # Nussinov features on the 11-nt context, center index = 5
      nuss_e, nuss_unp = nuss_features(ctx, center_index=5)

      key = f"{r['chrom']}:{pos}:{r['strand']}"
      label = 1 if (labels and key in labels) else 0

      rows.append({
        'chrom': r['chrom'],
        'pos': int(pos),
        'strand': r['strand'],
        'intron_start': int(r['start']),
        'intron_end': int(r['end']),
        'three_ss': int(r['three_ss']),
        'ctx11': ctx,
        'dist_to_3ss': int(dist),
        'polyY_frac': fracY,
        'polyY_longest': runY,
        'polyY_YYY3': y3,
        'u2_score': u2s,
        'u2_align_score': u2_align,
        'hmm_logp_ctx': hmm_logp,
        'hmm_pB_center': hmm_pB,
        'nuss_min_energy': nuss_e,
        'nuss_unpaired': nuss_unp,
        'label': label
      })
      seqs_for_vocab.append(ctx)

  df = pd.DataFrame(rows)
  if df.empty:
    raise RuntimeError("No candidates produced. Check inputs and coordinates.")

  vocab, index = make_kmer_vocab(df['ctx11'].values, k=KMER_K, top=256)
  X_kmer = np.vstack([onehot_kmers(s, index, k=KMER_K) for s in df['ctx11'].values])

  X_num = df[[
    'dist_to_3ss',
    'polyY_frac',
    'polyY_longest',
    'polyY_YYY3',
    'u2_score',
    'u2_align_score',
    'hmm_logp_ctx',
    'hmm_pB_center',
    'nuss_min_energy',
    'nuss_unpaired'
  ]].to_numpy(dtype=np.float32)

  X = np.hstack([X_num, X_kmer])
  feature_names = [
    'dist_to_3ss',
    'polyY_frac',
    'polyY_longest',
    'polyY_YYY3',
    'u2_score',
    'u2_align_score',
    'hmm_logp_ctx',
    'hmm_pB_center',
    'nuss_min_energy',
    'nuss_unpaired'
  ] + [f'kmer_{m}' for m in vocab]

  return df, X, feature_names, index

# ---------- Training / scoring ----------

def fit_and_eval(X, y, outdir):
  metrics = {}

  lr = Pipeline([
    ('sc', StandardScaler(with_mean=False)),
    ('clf', LogisticRegression(max_iter=200, class_weight='balanced', solver='lbfgs'))
  ])

  rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    n_jobs=-1,
    class_weight='balanced_subsample',
    random_state=1
  )

  mlp = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=50, random_state=1)

  models = {'logreg': lr, 'rf': rf, 'mlp': mlp}
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

  for name, model in models.items():
    ap_scores, roc_scores, f1_scores = [], [], []
    for tr, te in skf.split(X, y):
      model.fit(X[tr], y[tr])
      if hasattr(model, "predict_proba"):
        p = model.predict_proba(X[te])[:, 1]
      else:
        p = model.decision_function(X[te])
        p = (p - p.min()) / (p.max() - p.min() + 1e-9)
      yhat = (p >= 0.5).astype(int)
      try:
        ap_scores.append(average_precision_score(y[te], p))
      except Exception:
        ap_scores.append(float('nan'))
      roc_scores.append(roc_auc_score(y[te], p))
      f1_scores.append(f1_score(y[te], yhat))
    metrics[name] = {
      'AP_mean': float(np.nanmean(ap_scores)),
      'ROC_AUC_mean': float(np.mean(roc_scores)),
      'F1_mean': float(np.mean(f1_scores))
    }
    model.fit(X, y)
    dump(model, Path(outdir) / f"model_{name}.joblib")

  with open(Path(outdir) / "train_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
  return metrics

def score_with_model(X, model_path):
  from joblib import load
  model = load(model_path)
  if hasattr(model, "predict_proba"):
    return model.predict_proba(X)[:, 1]
  s = model.decision_function(X)
  s = (s - s.min()) / (s.max() - s.min() + 1e-9)
  return s

def best_per_intron(df):
  grp = df.groupby(['chrom', 'intron_start', 'intron_end', 'strand'])
  idx = grp['score'].idxmax()
  return df.loc[idx.values].sort_values(['chrom', 'intron_start'])

def write_tsv_gz(df, path):
  with gzip.open(path, 'wt') as f:
    df.to_csv(f, sep='\t', index=False)

def main():
  ap = argparse.ArgumentParser(description="Branchpoint prediction pipeline (Python)")
  ap.add_argument('--introns', required=True, help='TSV with columns: chrom,start,end,strand,three_ss')
  ap.add_argument('--fasta', required=True, help='Reference genome FASTA (with .fai index)')
  ap.add_argument('--labels', default=None, help='Optional TSV of known branchpoints: chrom,pos,strand')
  ap.add_argument('--outdir', required=True, help='Output directory')
  ap.add_argument('--model', default=None, help='Optional path to a fitted model_*.joblib for scoring')
  args = ap.parse_args()

  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)

  introns = read_introns(args.introns)
  fasta = Fasta(args.fasta)
  label_set = load_labels(args.labels) if args.labels else None

  df_cand, X, feat_names, kindex = build_dataset(introns, fasta, labels=label_set, outdir=outdir)
  write_tsv_gz(df_cand, outdir / "candidates.tsv.gz")

  if args.labels:
    y = df_cand['label'].to_numpy(dtype=np.int32)
    metrics = fit_and_eval(X, y, outdir)
    print("CV metrics:", json.dumps(metrics, indent=2))

  model_path = args.model if args.model else (outdir / "model_rf.joblib")
  if os.path.exists(model_path):
    scores = score_with_model(X, str(model_path))
    df_out = df_cand.copy()
    df_out['score'] = scores
    write_tsv_gz(df_out, outdir / "predictions.tsv.gz")
    df_best = best_per_intron(df_out)
    write_tsv_gz(df_best, outdir / "predictions_best_per_intron.tsv.gz")
    print(f"Wrote predictions to {outdir}")
  else:
    print("No model provided and model_rf.joblib not found; training or scoring skipped.")

if __name__ == '__main__':
  main()
