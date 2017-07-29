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
            ref_seq = ref_seq.upper()

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

            # 0-4,  ref base
            # 4-9,  call base (A, C, G, T, Del)
            # 9-12, het, hom
            # type, SNP, del, ins, none, other

            base_vec = [0,0,0,0,
                        0,0,0,0,0,
                        0,0,
                        0,0,0,0,0,0] 

            if len(row[1]) == 1 and len(row[2]) == 1:
                base_vec[base2num[row[1][0]]] = 1 
                base_vec[4 + base2num[row[2][0]]] = 1
                if het:
                    base_vec[9] = 1.
                    base_vec[11] = 1.
                else:
                    base_vec[10] = 1.
                    base_vec[11] = 1.

            elif len(row[1]) > 1 or len(row[2]) > 1 :  # not simple SNP case
                if len(row[1]) == 2 and len(row[2]) == 1: # simple deletion
                    base_vec[base2num[row[1][1]]] = 1 
                    base_vec[8] = 1
                    if het:
                        base_vec[9] = 1.
                        base_vec[12] = 1.
                    else:
                        base_vec[10] = 1.
                        base_vec[12] = 1.
                elif len(row[1]) == 1 and len(row[2]) == 2: # simple insertion
                    base_vec[base2num[row[1][0]]] = 1 
                    base_vec[4 + base2num[row[2][1]]] = 1
                    if het:
                        base_vec[9] = 1.
                        base_vec[13] = 1.
                    else:
                        base_vec[10] = 1.
                        base_vec[13] = 1.
                else: # complicated insertions and deletions
                    base_vec[base2num[row[1][0]]] = 1 
                    base_vec[8] = 1
                    if het:
                        base_vec[9] = 1.
                        if len(row[1]) > len(row[2]):
                            base_vec[15] = 1.
                        else:
                            base_vec[16] = 1.
                    else:
                        base_vec[10] = 1.
                        if len(row[1]) < len(row[2]):
                            base_vec[16] = 1.
                        else:
                            base_vec[15] = 1.
        
            Y_intitial[pos] = base_vec
    """ # if two SNPs are nearby, call the variant as "other" kind, comment out this for now 
    Y_pos = sorted(Y_intitial.keys())
    cpos = Y_pos[0]
    for pos in Y_pos[1:]:
        if abs(pos - cpos) < 12:
            Y_intitial[pos][15] = 1
            Y_intitial[cpos][15] = 1
            
            Y_intitial[pos][11] = 0
            Y_intitial[cpos][11] = 0
            Y_intitial[pos][12] = 0
            Y_intitial[cpos][12] = 0
            Y_intitial[pos][13] = 0
            Y_intitial[cpos][13] = 0
            Y_intitial[pos][14] = 0
            Y_intitial[cpos][14] = 0
        cpos = pos
    """
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
                # base_vec = [0,0,0,0,0,0,0,0]
                base_vec = [0,0,0,0,
                            0,0,0,0,0,
                            0,0,
                            0,0,0,0,0,0] 
                base_vec[base2num[ref_seq[7]]] = 1 
                base_vec[4 + base2num[ref_seq[7]]] = 1
                base_vec[10] = 1.
                base_vec[14] = 1.
                
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
