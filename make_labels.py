import sys
from collections import defaultdict
import bisect

def load_introns(intron_tsv_path):
  """
  Reads introns.tsv with columns:
  chrom  start  end  strand  three_ss
  Adds intron_id = intron_1, intron_2, ...
  Returns:
    introns: list of intron dicts
    index: {(chrom,strand): {"starts": [...], "introns": [...]} }
  """
  introns = []
  index = {}

  with open(intron_tsv_path) as f:
    header = f.readline().strip().split()
    # Expect: chrom start end strand three_ss
    for idx, line in enumerate(f, 1):
      if not line.strip():
        continue
      parts = line.strip().split()
      chrom = parts[0]
      start = int(parts[1])
      end = int(parts[2])
      strand = parts[3]
      three_ss = int(parts[4])

      intron_id = f"intron_{idx}"
      intron = {
        "id": intron_id,
        "chr": chrom,
        "start": start,
        "end": end,
        "strand": strand,
        "three_ss": three_ss,
      }
      introns.append(intron)

      key = (chrom, strand)
      if key not in index:
        index[key] = {"starts": [], "introns": []}
      index[key]["starts"].append(start)
      index[key]["introns"].append(intron)

  # sort each chrom/strand list by start
  for key in index:
    starts = index[key]["starts"]
    intron_list = index[key]["introns"]
    # sort by start, keeping introns aligned
    pairs = sorted(zip(starts, intron_list), key=lambda x: x[0])
    index[key]["starts"] = [p[0] for p in pairs]
    index[key]["introns"] = [p[1] for p in pairs]

  return introns, index

def assign_branchpoints(bp_bed_path, index):
  """
  Reads Mercer S2 bed file:
    chrom  bp_start  bp_end  name  score  strand
  For each BP, finds containing intron via index and
  records 1-based BP position per intron_id.
  Returns:
    bp_by_intron: {intron_id: [bp_pos_1based, ...]}
  """
  bp_by_intron = defaultdict(list)

  with open(bp_bed_path) as f:
    for line in f:
      if not line.strip():
        continue
      parts = line.strip().split()
      chrom = parts[0]
      bp_start = int(parts[1])  # 0-based
      strand = parts[5]

      key = (chrom, strand)
      if key not in index:
        continue

      starts = index[key]["starts"]
      introns = index[key]["introns"]

      # find rightmost intron start <= bp_start
      i = bisect.bisect_right(starts, bp_start) - 1
      if i < 0:
        continue

      intron = introns[i]
      if intron["start"] <= bp_start < intron["end"]:
        bp_pos_1based = bp_start + 1
        bp_by_intron[intron["id"]].append(bp_pos_1based)

  return bp_by_intron

def write_labels(out_path, introns, bp_by_intron):
  with open(out_path, "w") as out:
    out.write("intron_id\tchr\tstart\tend\tstrand\tthree_ss\tbp_positions_1based\n")
    for intron in introns:
      iid = intron["id"]
      positions = sorted(set(bp_by_intron.get(iid, [])))
      bp_str = ",".join(str(p) for p in positions)
      out.write(
        f"{iid}\t{intron['chr']}\t{intron['start']}\t{intron['end']}\t"
        f"{intron['strand']}\t{intron['three_ss']}\t{bp_str}\n"
      )

def main():
  if len(sys.argv) != 4:
    print("Usage: python make_labels_no_bedtools.py introns.tsv S2_branchpoints.bed labels.tsv")
    sys.exit(1)

  intron_tsv = sys.argv[1]
  bp_bed = sys.argv[2]
  out_tsv = sys.argv[3]

  introns, index = load_introns(intron_tsv)
  bp_by_intron = assign_branchpoints(bp_bed, index)
  write_labels(out_tsv, introns, bp_by_intron)

if __name__ == "__main__":
  main()
