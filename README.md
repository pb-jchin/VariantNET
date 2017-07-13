
VariantNET
===============================

In this repository, we demostrate using a simple convolution neural network (implented 
with Tensorflow) for variant calling.  This is sepecially useful for DNA sequences with 
many insertion and deletion errors (e.g. from the single molecule DNA sequencing with 
PacBio's platform). With insertion and deletion errors, we will see more alignment slippage,
even with SNP variants, there are some possibilities that the useful information is not precisely
at the column of SNP location in the alignments.  With a CNN, it is possible to aggregate 
information from nearby bases so it can outperform simple pile-up counting for variant
calling. This is a simple test to see how well we can do it without signal level information.
With the singal level information and a better alignment model, it is possible to further
improve the performance.

July 13 2017

Jason Chin
