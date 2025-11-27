import sys

def parse_gtf(path):
  transcripts = {}
  with open(path) as f:
    for line in f:
      if line.startswith("#"):
        continue
      parts = line.strip().split("\t")
      if parts[2] != "exon":
        continue

      chrom = parts[0]
      start = int(parts[3]) - 1  # convert to 0-based
      end = int(parts[4])
      strand = parts[6]

      # extract transcript_id
      attrs = parts[8]
      tid = None
      for field in attrs.split(";"):
        field = field.strip()
        if field.startswith("transcript_id"):
          tid = field.split("\"")[1]
          break
      if tid is None:
        continue

      if tid not in transcripts:
        transcripts[tid] = {
          "chrom": chrom,
          "strand": strand,
          "exons": []
        }
      transcripts[tid]["exons"].append((start, end))

  return transcripts

def extract_introns(transcripts, out_path):
  with open(out_path, "w") as out:
    out.write("chrom\tstart\tend\tstrand\tthree_ss\n")

    for tid, data in transcripts.items():
      chrom = data["chrom"]
      strand = data["strand"]
      exons = sorted(data["exons"], key=lambda x: x[0])

      # introns = gaps between exons
      for i in range(len(exons) - 1):
        e1_end = exons[i][1]
        e2_start = exons[i + 1][0]

        # intron coordinates
        intron_start = e1_end
        intron_end = e2_start

        if intron_end <= intron_start:
          continue  # malformed or overlapping exons

        # compute 3' splice site (depends on strand)
        if strand == "+":
          three_ss = intron_end  # 3'SS is at the end of the intron
        else:
          three_ss = intron_start  # 3'SS is at the start of the intron

        out.write(f"{chrom}\t{intron_start}\t{intron_end}\t{strand}\t{three_ss}\n")

def main():
  if len(sys.argv) != 3:
    print("Usage: python create_introns_from_gtf.py gencode.gtf introns.tsv")
    sys.exit(1)

  gtf_in = sys.argv[1]
  introns_out = sys.argv[2]

  transcripts = parse_gtf(gtf_in)
  extract_introns(transcripts, introns_out)

if __name__ == "__main__":
  main()
