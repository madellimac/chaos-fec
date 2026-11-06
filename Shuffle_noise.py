#!/usr/bin/env python3

import numpy as np
import random 
import math
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../py_aff3ct/build/lib')
import py_aff3ct as aff3ct

from factory          import factory

def generate(code_type, N, K, EbN0, I, shift):

	fac = factory(code_type, N, K, EbN0, I)

	vec_src    = np.ndarray(K,  dtype = np.int32)
	r_in       = np.ndarray(N,  dtype = np.float32)

	with open('noisy_cw.npy', 'rb') as f:
		r_in = np.load(f)
		vec_src = np.load(f)

	mag_r_in = np.ndarray(N,  dtype = np.float32)
	s_r_in = np.ndarray(N,  dtype = np.float32)

	s_r_in = np.sign(r_in)
	mag_r_in[:] = np.abs(r_in)
	#mag_r_in = np.roll(mag_r_in, shift)

	np.random.shuffle(mag_r_in)
	rand_ix1 = np.random.randint(N)
	mag_r_in[rand_ix1] = -5*mag_r_in[rand_ix1]
	rand_ix2 = np.random.randint(N)
	mag_r_in[rand_ix2] = -5*mag_r_in[rand_ix2]
	rand_ix3 = np.random.randint(N)
	mag_r_in[rand_ix3] = -5*mag_r_in[rand_ix3]

	r_in = s_r_in * mag_r_in
	enc, dec2   = fac.build()
	
	mnt2   = aff3ct.module.monitor.Monitor_BFER(K, 1)

	dec2["decode_siho   :: V_K  "] = mnt2  ["check_errors2::V"   ]
	dec2["decode_siho  ::Y_N" ].bind(r_in)
	mnt2["check_errors2:: U    " ].bind(vec_src)
	
	dec2['decode_siho'].exec()
	mnt2['check_errors2'].exec()
	
	#diff = dec2["decode_siho   :: V_K  "][:] - (-2147483648*mnt2["check_errors2:: U    " ][:])
	diff = np.abs(dec2["decode_siho   :: V_K  "][:] - mnt2["check_errors2:: U    " ][:])
	# print(dec2["decode_siho   :: V_K  "][:])
	# print(mnt2["check_errors2:: U    " ][:])
	error_position = np.nonzero(diff[0])

	error_vector = np.zeros(N,  dtype = np.int32)

	for pos in error_position:
		error_vector[pos] = 1

	return(error_vector)

	#print("be=",mnt2.get_n_be())
	#print("fe=",mnt2.get_n_fe())
    
	#return (mnt2.get_n_be())
	## Il semblerait qu'il y ai un offset de -1 selon x