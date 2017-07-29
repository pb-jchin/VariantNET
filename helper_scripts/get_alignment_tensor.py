
from falcon_kit.FastaReader import FastaReader
from collections import Counter
import argparse
import logging
import os
import re
import shlex
import subprocess
import sys
import numpy as np
from math import log







cigar_re = r"(\d+)([MIDNSHP=X])"
base2num = dict(zip("ACGT", (0,1,2,3)))

def generate_aln_count_tensor(alns, center, ref_seq):
    aln_code = np.zeros( (15, 3, 4) )
    for aln in alns:
        for rp, qp, rb, qb in aln:
            if qb not in ("A","C","G","T","-"):
                continue
            rb = rb.upper()
            if rb not in ("A","C","G","T","-"):
                continue
            if rp - center >= -8 and rp - center < 7:
                offset = rp - center + 8
                if rb != "-":
                    aln_code[offset][0][ base2num[rb] ] += 1
                    if qb != "-":
                        aln_code[offset][1][ base2num[qb] ] += 1
                        aln_code[offset][2][ base2num[qb] ] += 1
                else:
                    aln_code[offset][1][ base2num[qb] ] += 1
    output_line = []
    output_line.append( "%d %s" %  (center, ref_seq[center-8:center+7]) )
    for c1 in np.reshape(aln_code, 15*3*4):
        output_line.append("%0.1f" % c1)
    return " ".join(output_line)

def output_aln_tensor(args):

    bam_file_fn = args.bam_file_fn
    pm_count_fn = args.pm_count_fn
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


    begin2end = {}
    with open(pm_count_fn) as f:
        for row in f.readlines():
            row = row.strip().split()
            pos = int(row[0])
            begin2end[ pos-8 ] = (pos + 8, pos)

    # maybe we should check if the samtools path is valid
    p = subprocess.Popen(shlex.split("%s view %s" % (samtools, bam_file_fn) ), stdout=subprocess.PIPE)

    center_to_aln = {}

    for l in p.stdout:
        l = l.strip().split()
        if l[0][0] == "@":
            continue

        QNAME = l[0]
        FLAG = int(l[1])
        if FLAG != 0 and FLAG != 16:
            continue
        RNAME = l[2]
        POS = int(l[3]) - 1 #make it zero base to match sequence index
        CIGAR = l[5]
        SEQ = l[9]
        rp = POS
        qp = 0

        end_to_center = {}
        active_set = set()

        for m in re.finditer(cigar_re, CIGAR):
            adv = int(m.group(1))
            if m.group(2) == "S":
                qp += adv
            if m.group(2) in ("M", "=", "X"):
                matches = []
                for i in xrange(adv):
                    matches.append( (rp, SEQ[qp]) )

                    if rp in begin2end:
                        r_end, r_center = begin2end[rp]
                        end_to_center[r_end] = r_center
                        active_set.add(r_center)
                        center_to_aln.setdefault(r_center, [])
                        center_to_aln[r_center].append([])

                    for center in list(active_set):
                        center_to_aln[center][-1].append( (rp, qp, ref_seq[rp], SEQ[qp] ) )

                    if rp in end_to_center:
                        center = end_to_center[rp]
                        active_set.remove(center)

                    rp += 1
                    qp += 1

            elif m.group(2) == "I":
                for i in xrange(adv):
                    if i < 4: #  ignore extra bases if the insert is too long
                        for center in list(active_set):
                            center_to_aln[center][-1].append( (rp, qp, "-", SEQ[qp] ))

                    qp += 1

            elif m.group(2) == "D":
                for i in xrange(adv):
                    for center in list(active_set):
                        center_to_aln[center][-1].append( (rp, qp, ref_seq[rp], "-" ))

                    if rp in begin2end:
                        r_end, r_center = begin2end[rp]
                        end_to_center[r_end] = r_center
                        active_set.add(r_center)
                        center_to_aln.setdefault(r_center, [])
                        center_to_aln[r_center].append([])

                    if rp in end_to_center:
                        center = end_to_center[rp]
                        active_set.remove(center)

                    rp += 1


        for center in center_to_aln.keys():
            if center + 8 < POS:
                t_line =  generate_aln_count_tensor(center_to_aln[center], center, ref_seq)
                print t_line
                del center_to_aln[center]

    for center in center_to_aln.keys():
        if center + 8 < POS:
            t_line =  generate_aln_count_tensor(center_to_aln[center], center, ref_seq)
            print t_line


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
      description='Generate a 15x4x3 "tensor" summarizing local alignments from a BAM file and a list of candidate locations' )

    parser.add_argument('--bam_file_fn', type=str, default="input.bam", 
            help="path to the sorted bam file that contains the alignments, default:input.bam")

    parser.add_argument('--pm_count_fn', type=str, default="pm_counts", 
            help="pile-up count input, default:pm_count")
    
    parser.add_argument('--ref_fasta_fn', type=str, default="ref.fa", 
            help="path to the reference fasta file, default:ref.fa")


    parser.add_argument('--ctg_name', type=str, default="ctg", 
            help="the reference name, defaults:ctg")

    parser.add_argument('--samtools', type=str, default="samtools", 
            help="the path to `samtools` command, default:samtools")

    args = parser.parse_args()

    output_aln_tensor(args)

