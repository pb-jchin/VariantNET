import intervaltree
import numpy as np
import random

def get_batch(X, Y, size=100):
    s = random.randint(0,len(X)-size)
    return X[s:s+size], Y[s:s+size]

def get_aln_array( aln_tensor_fn ):

    X_intitial = {}  

    with open( aln_tensor_fn ) as f:
        for row in f:
            row = row.strip().split()
            pos = int(row[0])
            ref_seq = row[1]

            if ref_seq[7] not in ["A","C","G","T"]:
                continue

            vec = np.reshape(np.array([float(x) for x in row[2:]]), (15,3,4))

            vec = np.transpose(vec, axes=(0,2,1))
            if sum(vec[7,:,0]) < 5:
                continue
            
            vec[:,:,1] -= vec[:,:,0]
            vec[:,:,2] -= vec[:,:,0]
            
            X_intitial[pos] = vec
                
    all_pos = sorted(X_intitial.keys())

    Xarray = []
    pos_array = []
    for pos in all_pos:
        Xarray.append(X_intitial[pos])
        pos_array.append(pos)
    Xarray = np.array(Xarray)

    return Xarray, pos_array

def get_training_array( aln_tensor_fn, variant_set_fn, mask_bed_fn ):
    base2num = dict(zip("ACGT",(0, 1, 2, 3)))

    tree =  intervaltree.IntervalTree()
    with open(mask_bed_fn) as f:
        for row in f:
            row = row.strip().split()
            b = int(row[1])
            e = int(row[2])
            tree.addi(b, e, None)

    Y_intitial = {}
    with open( variant_set_fn ) as f:
        for row in f:
            row = row.strip().split()
            if row[3] == "0":
                het = True
            else:
                het = False
            
            pos = int(row[0])
            if len(tree.search(pos)) == 0:
                continue
            base_vec = [0,0,0,0,0,0,0,0]  #first 4, base vec, last 4, het, hom, non-variant, not-SNPs
            if het:
                base_vec[base2num[row[1][0]]] = 0.5
                base_vec[base2num[row[2][0]]] = 0.5
                base_vec[4] = 1.
            else:
                base_vec[base2num[row[2][0]]] = 1
                base_vec[5] = 1.

            if len(row[1]) > 1 or len(row[2]) > 1 :  # not simple SNP case
                base_vec[7] = 1.
                base_vec[4] = 0.
                base_vec[5] = 0.
        
            Y_intitial[pos] = base_vec
            
    Y_pos = sorted(Y_intitial.keys())
    cpos = Y_pos[0]
    for pos in Y_pos[1:]:
        if abs(pos - cpos) < 12:
            Y_intitial[pos][7] = 1
            Y_intitial[cpos][7] = 1
            
            Y_intitial[pos][4] = 0
            Y_intitial[cpos][4] = 0
            Y_intitial[pos][5] = 0
            Y_intitial[cpos][5] = 0
        cpos = pos

    X_intitial = {}  

    with open( aln_tensor_fn ) as f:
        for row in f:
            row = row.strip().split()
            pos = int(row[0])
            if len(tree.search(pos)) == 0:
                continue
            ref_seq = row[1]
            if ref_seq[7] not in ["A","C","G","T"]:
                continue
            vec = np.reshape(np.array([float(x) for x in row[2:]]), (15,3,4))

            vec = np.transpose(vec, axes=(0,2,1))
            if sum(vec[7,:,0]) < 5:
                continue
            
            vec[:,:,1] -= vec[:,:,0]
            vec[:,:,2] -= vec[:,:,0]

            
            X_intitial[pos] = vec
            
            if pos not in Y_intitial:
                base_vec = [0,0,0,0,0,0,0,0]
                base_vec[base2num[ref_seq[7]]] = 1
                base_vec[6] = 1.
                Y_intitial[pos] = base_vec
                
    all_pos = sorted(X_intitial.keys())
    random.shuffle(all_pos)

    Xarray = []
    Yarray = []
    pos_array = []
    for pos in all_pos:
        Xarray.append(X_intitial[pos])
        Yarray.append(Y_intitial[pos])
        pos_array.append(pos)
    Xarray = np.array(Xarray)
    Yarray = np.array(Yarray)

    return Xarray, Yarray, pos_array
