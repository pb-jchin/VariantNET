python ../helper_scripts/get_SNP_candidates.py --bam_file_fn ../testing_data/chr21/hg38.NA12878-WashU_chr21-14069662-46411975.bam \
                                               --ref_fasta_fn ../testing_data/chr21/chr21.fa \
                                               --pm_count_fn pm_counts_chr21 --ctg_nam chr21 
python ../helper_scripts/get_alignment_tensor.py --bam_file_fn ../testing_data/chr21/hg38.NA12878-WashU_chr21-14069662-46411975.bam \
                                                 --pm_count_fn pm_counts_chr21 \
                                                 --ctg_name chr21 \
                                                 --ref_fasta_fn ../testing_data/chr21/chr21.fa > aln_tensor_chr21
 
python ../helper_scripts/get_variant_set.py < ../testing_data/chr21/chr21.vcf > variants_chr21
