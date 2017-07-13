python ../helper_scripts/get_SNP_candidates.py --bam_file_fn ../testing_data/chr22/hg38.NA12878-WashU_chr22-18924717-49973797.bam \
                                               --ref_fasta_fn ../testing_data/chr22/chr22.fa \
                                               --pm_count_fn pm_counts_chr22 --ctg_nam chr22 
python ../helper_scripts/get_alignment_tensor.py --bam_file_fn ../testing_data/chr22/hg38.NA12878-WashU_chr22-18924717-49973797.bam \
                                                 --pm_count_fn pm_counts_chr22 \
                                                 --ref_fasta_fn ../testing_data/chr22/chr22.fa \
                                                 --ctg_name chr22 > aln_tensor_chr22
python ../helper_scripts/get_variant_set.py < ../testing_data/chr22/chr22.vcf > variants_chr22
