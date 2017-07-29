from falcon_kit.FastaReader import FastaReader
import argparse
import os
import re
import shlex
import subprocess
import sys
from math import log


cigar_re = r"(\d+)([MIDNSHP=X])"

def output_count(pos, base_count, ref_base, min_cov, th):

    total_count = 0
    total_count += sum(c[1] for c in base_count) 
    if total_count < min_cov:
        return None



    base_count.sort(key = lambda x:-x[1])
    p0 = 1.0 *  base_count[0][1] / total_count
    p1 = 1.0 *  base_count[1][1] / total_count
    output_line = []
    if (p0 < 1.0 - th and p1 > th) or base_count[0][0] != ref_base:
        if base_count[1][0] == "I" or base_count[1][0] == "D":
            if p1 < 0.25: # hard-coded rule for indel for now, 0.25 is choosen for balance recall and PPV
                return None
        output_line = [pos+1, ref_base, total_count]
        output_line.extend( ["%s %d" % x for x in base_count] )
        output_line = " ".join([str(c) for c in output_line])
        return total_count, output_line
    else:
        return None

def make_variant_candidates( args ):

    bam_file_fn = args.bam_file_fn
    pm_count_fn = args.pm_count_fn
    threshold = args.threshold
    min_cov = args.min_cov
    ctg_name = args.ctg_name
    samtools = args.samtools
    ref_fasta_fn = args.ref_fasta_fn

    # assume the ref.fa has only one reference, the name does not mattere, we only read the first one
    ref_seq = None
    for r in FastaReader(ref_fasta_fn):
        if r.name != ctg_name:
            continue
        ref_seq = r.sequence

    if ref_seq == None:
        print >> sys.stderr, "Can't get reference sequence"
        sys.exit(1)

    # maybe we should check if the samtools path is valid
    p = subprocess.Popen(shlex.split("%s view %s" % (samtools, bam_file_fn ) ), stdout=subprocess.PIPE)
    pileup = {}

    pm_count_f = open(pm_count_fn, "w")

    for l in p.stdout:
        l = l.strip().split()
        if l[0][0] == "@":
            continue

        QNAME = l[0]
        RNAME = l[2]

        if RNAME != ctg_name:
            continue

        FLAG = int(l[1])
        if FLAG != 0 and FLAG != 16:
            continue
        POS = int(l[3]) - 1 #make it zero base to match sequence index 
        CIGAR = l[5]
        SEQ = l[9]
        rp = POS
        qp = 0

        skip_base = 0
        total_aln_pos = 0
        for m in re.finditer(cigar_re, CIGAR):
            adv = int(m.group(1))
            total_aln_pos += adv
            if m.group(2)  == "S":
                skip_base += adv

        if 1.0 - 1.0 * skip_base / (total_aln_pos+1) < 0.50: #if a read is less than 50% aligned, skip 
            continue

        for m in re.finditer(cigar_re, CIGAR):

            adv = int(m.group(1))

            if m.group(2) == "S":
                qp += adv

            if m.group(2) in ("M", "=", "X"):
                matches = []
                for i in range(adv):
                    matches.append( (rp, SEQ[qp]) )
                    rp += 1
                    qp += 1
                for pos, b in matches:
                    pileup.setdefault(pos, {"A":0, "C":0, "G":0, "T":0, "I":0, "D":0})
                    if b not in ["A","C","G","T"]:
                        continue
                    pileup[pos][b] += 1
            elif m.group(2) == "I":
                pileup.setdefault(rp, {"A":0, "C":0, "G":0, "T":0, "I":0, "D":0})
                pileup[rp]["I"] += 1
                for i in xrange(adv):
                    qp += 1
            elif m.group(2) == "D":
                for i in xrange(adv):
                    if adv < 3: # ignore large deletions with mostly caused be SVs
                        pileup.setdefault(rp, {"A":0, "C":0, "G":0, "T":0, "I":0, "D":0})
                        pileup[rp]["D"] += 1 
                    rp += 1

        pos_k = pileup.keys()
        pos_k.sort()

        th = threshold
        for pos in pos_k:
            if pos < POS:  # output pileup informaiton before POS which is the current head of the ref 
                base_count = pileup[pos].items()
                ref_base = ref_seq[pos].upper()
                out = output_count(pos, base_count, ref_base, min_cov, th)
                if out != None:
                    total_count, out_line = out
                    print >> pm_count_f, out_line

                del pileup[pos]

    # for the last one
    th = threshold
    pos_k = pileup.keys()
    pos_k.sort()
    for pos in pos_k:
        base_count = pileup[pos].items()
        ref_base = ref_seq[pos].upper()
        out = output_count(pos, base_count, ref_base, min_cov, th)
        if out != None:
            total_count, out_line = out
            print >> pm_count_f, out_line

        del pileup[pos]



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Generate SNP candidates using alignment pile-up')

    parser.add_argument('--bam_file_fn', type=str, default="input.bam", 
            help="path to the sorted bam file that contains the alignments, default:input.bam")

    parser.add_argument('--pm_count_fn', type=str, default="pm_count", 
            help="pile-up count output, default:pm_count")
    
    parser.add_argument('--ref_fasta_fn', type=str, default="ref.fa", 
            help="path to the reference fasta file, default:ref.fa")

    parser.add_argument('--threshold', type=float, default=0.15, 
            help="minimum frequence threshold for 2nd allele to be considered as a condidate site, default:0.15")

    parser.add_argument('--min_cov', type=float, default=10, 
            help="minimum coverage for making a variant call, default=10")

    parser.add_argument('--ctg_name', type=str, default="ctg", 
            help="the reference name, defaults:ctg")

    parser.add_argument('--samtools', type=str, default="samtools", 
            help="the path to `samtools` command, default:samtools")



    args = parser.parse_args()

    make_variant_candidates(args)

