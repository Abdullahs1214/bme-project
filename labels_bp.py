import sys

def main():
  if len(sys.argv) != 3:
    print("Usage: python labels_bp.py labels.tsv labels_bp.tsv")
    sys.exit(1)

  labels_in = sys.argv[1]
  labels_out = sys.argv[2]

  with open(labels_in) as f_in, open(labels_out, "w") as f_out:
    # write header for BP.py
    f_out.write("chrom\tpos\tstrand\n")

    # skip header line of labels.tsv
    header = f_in.readline()

    for line in f_in:
      line = line.strip()
      if not line:
        continue

      parts = line.split("\t")

      # We expect at least: intron_id, chr, start, end, strand, three_ss
      if len(parts) < 6:
        continue  # malformed line, skip

      intron_id = parts[0]
      chrom = parts[1]
      # start = parts[2]
      # end = parts[3]
      strand = parts[4]

      # bp_positions_1based is the 7th column if present
      bp_list = parts[6] if len(parts) >= 7 else ""

      if not bp_list.strip():
        continue  # no branchpoints for this intron

      for bp in bp_list.split(","):
        bp = bp.strip()
        if not bp:
          continue
        f_out.write(f"{chrom}\t{bp}\t{strand}\n")

if __name__ == "__main__":
  main()
