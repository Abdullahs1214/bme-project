"""
Branchpoint prediction pipeline (Python, LazyPredict)
"""

import argparse, os, json, gzip
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from joblib import dump
from pyfaidx import Fasta as FaidxFasta

DNA = set("ACGTN")
KMER_K = 3   # k-mer size for sequence context
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
  seq = seq.upper()
  n = len(seq)
  if n == 0:
    return 0.0, []

  M = [[0.0] * n for _ in range(n)]
  trace = [[None] * n for _ in range(n)]

  for length in range(1, n):
    for i in range(n - length):
      j = i + length

      opt_i_unpaired = M[i + 1][j] + (0.1 if i != 0 else 0.0)
      best = opt_i_unpaired
      trace[i][j] = ('i_unpaired', i + 1, j)

      opt_j_unpaired = M[i][j - 1] + (0.1 if j != n - 1 else 0.0)
      if opt_j_unpaired < best:
        best = opt_j_unpaired
        trace[i][j] = ('j_unpaired', i, j - 1)

      e = nuss_energy(seq[i], seq[j])
      if e is not None and (j - i - 1) >= 3:
        opt_pair = M[i + 1][j - 1] + e
        if opt_pair < best:
          best = opt_pair
          trace[i][j] = ('pair', i + 1, j - 1)

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
  if not ctx:
    return 0.0, 1
  e, paired = nussinov_min_energy_window(ctx)
  if not paired or center_index >= len(paired):
    return float(e), 1
  is_unpaired = 1 if paired[center_index] == -1 else 0
  return float(e), int(is_unpaired)

# ---------- HMM helpers ----------

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

# ---------- Core helpers ----------

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

def read_labels(path):
  if path is None:
    return None
  df = pd.read_csv(path, sep='\t', header=0)
  needed = {'chrom', 'pos', 'strand'}
  missing = needed - set(df.columns)
  if missing:
    raise ValueError(f"Missing columns in labels file: {missing}")
  df = df.copy()
  df['pos'] = df['pos'].astype(int)
  return df

def build_label_index(df_labels):
  if df_labels is None:
    return set()
  return set(zip(df_labels['chrom'], df_labels['pos'], df_labels['strand']))

def polyY_stats(seq):
  if not seq:
    return 0.0, 0, 0
  seq = seq.upper()
  isY = np.array([1 if c in ('C', 'T') else 0 for c in seq], dtype=int)
  frac = isY.mean() if len(seq) > 0 else 0.0
  longest = 0
  current = 0
  for v in isY:
    if v == 1:
      current += 1
      longest = max(longest, current)
    else:
      current = 0
  if len(seq) >= 3:
    tail3 = seq[-3:]
    yyy3 = 1 if all(c in ('C', 'T') for c in tail3) else 0
  else:
    yyy3 = 0
  return frac, longest, yyy3

def u2_duplex_score(branch_ctx11):
  if len(branch_ctx11) != 11:
    return 0.0
  s = branch_ctx11.upper()
  score = 0
  if s[5] == 'A':
    score += 1
  if s[4] in ('C', 'T'):
    score += 0.5
  if s[6] in ('C', 'T'):
    score += 0.5
  if s[5] == 'G':
    score -= 1
  return score

def make_kmer_vocab(seqs, k=3, top=256):
  counts = Counter()
  for s in seqs:
    s = s.upper()
    for i in range(len(s) - k + 1):
      kmer = s[i:i + k]
      if all(c in DNA for c in kmer):
        counts[kmer] += 1
  vocab = [k for k, _ in counts.most_common(top)]
  index = {k: i for i, k in enumerate(vocab)}
  return vocab, index

def onehot_kmers(seq, index, k=3):
  vec = np.zeros(len(index), dtype=np.float32)
  seq = seq.upper()
  for i in range(len(seq) - k + 1):
    kmer = seq[i:i + k]
    if kmer in index:
      vec[index[kmer]] += 1.0
  if vec.sum() > 0:
    vec /= vec.sum()
  return vec

# ---------- Candidate generation ----------

def generate_candidates(df_introns, fa):
  rows = []
  for _, row in df_introns.iterrows():
    chrom = row['chrom']
    start = int(row['start'])
    end = int(row['end'])
    strand = row['strand']
    three_ss = int(row['three_ss'])

    if strand == '+':
      intron_start = start
      intron_end = end
      win_start = three_ss - 60
      win_end = three_ss - 10
      if win_end <= win_start:
        continue
      seq = fa.fetch(chrom, win_start, three_ss, strand='+')
      for i in range(len(seq)):
        gpos = win_start + i + 1
        base = seq[i]
        if base != 'A':
          continue
        ctx_start = max(0, i - SEQ_WIN)
        ctx_end = min(len(seq), i + SEQ_WIN + 1)
        ctx = seq[ctx_start:ctx_end]
        if len(ctx) < 2 * SEQ_WIN + 1:
          pad_left = SEQ_WIN - (i - ctx_start)
          pad_right = (SEQ_WIN + 1) - (ctx_end - i)
          ctx = ('N' * pad_left) + ctx + ('N' * pad_right)
        dist_to_3ss = three_ss - gpos
        poly_start = three_ss - 50
        poly_start = max(poly_start, intron_start)
        poly_seq = fa.fetch(chrom, poly_start, three_ss, strand='+')
        py_frac, py_long, yyy3 = polyY_stats(poly_seq)
        u2_score = u2_duplex_score(ctx)
        u2_align = u2_alignment_score(ctx)
        hmm_logp, hmm_pB = hmm_branch_features(ctx)
        nuss_e, nuss_unp = nuss_features(ctx, center_index=SEQ_WIN)

        rows.append({
          'chrom': chrom,
          'intron_start': intron_start,
          'intron_end': intron_end,
          'strand': strand,
          'bp_pos': gpos,
          'three_ss': three_ss,
          'dist_to_3ss': dist_to_3ss,
          'ctx11': ctx,
          'polyY_frac': py_frac,
          'polyY_longest': py_long,
          'polyY_YYY3': yyy3,
          'u2_score': u2_score,
          'u2_align_score': u2_align,
          'hmm_logp_ctx': hmm_logp,
          'hmm_pB_center': hmm_pB,
          'nuss_min_energy': nuss_e,
          'nuss_unpaired': nuss_unp
        })

    else:
      intron_start = end
      intron_end = start
      win_start = three_ss + 10
      win_end = three_ss + 60
      if win_end <= win_start:
        continue
      seq = fa.fetch(chrom, three_ss, win_end, strand='-')
      for i in range(len(seq)):
        gpos = win_start - i
        base = seq[i]
        if base != 'A':
          continue
        ctx_start = max(0, i - SEQ_WIN)
        ctx_end = min(len(seq), i + SEQ_WIN + 1)
        ctx = seq[ctx_start:ctx_end]
        if len(ctx) < 2 * SEQ_WIN + 1:
          pad_left = SEQ_WIN - (i - ctx_start)
          pad_right = (SEQ_WIN + 1) - (ctx_end - i)
          ctx = ('N' * pad_left) + ctx + ('N' * pad_right)
        dist_to_3ss = gpos - three_ss
        poly_start = three_ss
        poly_end = three_ss + 50
        poly_seq = fa.fetch(chrom, poly_start, poly_end, strand='-')
        py_frac, py_long, yyy3 = polyY_stats(poly_seq)
        u2_score = u2_duplex_score(ctx)
        u2_align = u2_alignment_score(ctx)
        hmm_logp, hmm_pB = hmm_branch_features(ctx)
        nuss_e, nuss_unp = nuss_features(ctx, center_index=SEQ_WIN)

        rows.append({
          'chrom': chrom,
          'intron_start': intron_start,
          'intron_end': intron_end,
          'strand': strand,
          'bp_pos': gpos,
          'three_ss': three_ss,
          'dist_to_3ss': dist_to_3ss,
          'ctx11': ctx,
          'polyY_frac': py_frac,
          'polyY_longest': py_long,
          'polyY_YYY3': yyy3,
          'u2_score': u2_score,
          'u2_align_score': u2_align,
          'hmm_logp_ctx': hmm_logp,
          'hmm_pB_center': hmm_pB,
          'nuss_min_energy': nuss_e,
          'nuss_unpaired': nuss_unp
        })

  return pd.DataFrame(rows)

def attach_labels(df_cand, label_index):
  if not label_index:
    df = df_cand.copy()
    df['label'] = 0
    return df
  labels = []
  for _, row in df_cand.iterrows():
    key = (row['chrom'], int(row['bp_pos']), row['strand'])
    labels.append(1 if key in label_index else 0)
  df = df_cand.copy()
  df['label'] = labels
  return df

# ---------- Design matrix ----------

def make_design_matrix(df):
  df = df.copy()
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
  outdir = Path(outdir)
  outdir.mkdir(parents=True, exist_ok=True)

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
  )

  clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
  models_df, predictions = clf.fit(X_train, X_test, y_train, y_test)

  models_df.to_csv(outdir / "lazypredict_models.tsv", sep='\t')

  wanted = ["Accuracy", "F1 Score", "ROC AUC", "Time Taken"]
  cols_present = [c for c in wanted if c in models_df.columns]
  trimmed = models_df[cols_present].copy()
  metrics = trimmed.to_dict(orient='index')
  with open(outdir / "train_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

  best_name = None

  if "ROC AUC" in trimmed.columns:
    valid_roc = trimmed[trimmed["ROC AUC"].notna()]
    if not valid_roc.empty:
      best_name = valid_roc["ROC AUC"].idxmax()

  if best_name is None and "Accuracy" in trimmed.columns:
    valid_acc = trimmed[trimmed["Accuracy"].notna()]
    if not valid_acc.empty:
      best_name = valid_acc["Accuracy"].idxmax()

  if best_name is None:
    best_name = trimmed.index[0]

  best_model = clf.models[best_name]
  best_model.fit(X, y)
  safe_name = str(best_name).replace(" ", "_").lower()
  dump(best_model, outdir / f"model_{safe_name}.joblib")

  return metrics

def score_with_model(X, model_path):
  from joblib import load
  model = load(model_path)

  if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
      return proba[:, 1]
    if proba.shape[1] == 1:
      return 1.0 - proba[:, 0]

  if hasattr(model, "decision_function"):
    s = model.decision_function(X)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    return s

  pred = model.predict(X)
  return pred.astype(float)

def best_per_intron(df):
  grp = df.groupby(['chrom', 'intron_start', 'intron_end', 'strand'])
  idx = grp['score'].idxmax()
  return df.loc[idx.values].sort_values(['chrom', 'intron_start'])

def write_tsv_gz(df, path):
  with gzip.open(path, 'wt') as f:
    df.to_csv(f, sep='\t', index=False)

def main():
  ap = argparse.ArgumentParser(description="Branchpoint prediction pipeline (Python, LazyPredict)")
  ap.add_argument('--introns', required=True, help='TSV with columns: chrom,start,end,strand,three_ss')
  ap.add_argument('--fasta', required=True, help='Reference genome FASTA (with .fai index)')
  ap.add_argument('--labels', help='Optional TSV of known BPs (chrom,pos,strand)')
  ap.add_argument('--outdir', required=True, help='Output directory')
  ap.add_argument('--no-train', action='store_true', help='Skip training even if labels are provided')
  ap.add_argument('--model', help='Path to a trained model_*.joblib to use for scoring')
  args = ap.parse_args()

  outdir = Path(args.outdir)
  outdir.mkdir(parents=True, exist_ok=True)

  print("Reading introns...")
  df_introns = read_introns(args.introns)
  print(f"Loaded {len(df_introns)} introns.")

  print("Opening FASTA...")
  fa = Fasta(args.fasta)

  print("Generating candidate branchpoints...")
  df_cand = generate_candidates(df_introns, fa)
  print(f"Generated {len(df_cand)} candidate sites.")

  if args.labels:
    print("Reading labels...")
    df_labels = read_labels(args.labels)
    label_index = build_label_index(df_labels)
    print(f"Loaded {len(label_index)} labeled branchpoints.")
    df_cand = attach_labels(df_cand, label_index)
  else:
    df_cand = attach_labels(df_cand, None)

  print("Building design matrix...")
  df_design, X, feature_names, kmer_index = make_design_matrix(df_cand)
  write_tsv_gz(df_design, outdir / "candidates.tsv.gz")

  if args.labels and not args.no_train:
    print("Training models with LazyPredict...")
    y = df_design['label'].to_numpy(dtype=int)
    metrics = fit_and_eval(X, y, outdir)
    print("Training complete.")

  model_path = args.model
  if model_path is None:
    candidates = sorted(outdir.glob("model_*.joblib"))
    if candidates:
      model_path = candidates[0]

  if model_path and os.path.exists(model_path):
    print(f"Scoring candidates with model: {model_path}")
    scores = score_with_model(X, str(model_path))
    df_out = df_design.copy()
    df_out['score'] = scores
    write_tsv_gz(df_out, outdir / "predictions.tsv.gz")
    df_best = best_per_intron(df_out)
    write_tsv_gz(df_best, outdir / "predictions_best_per_intron.tsv.gz")
    print(f"Wrote predictions to {outdir}")
  else:
    print("No model provided and no model_*.joblib found; training or scoring skipped.")

if __name__ == '__main__':
  main()
