#!/usr/bin/env python3

import numpy as np
import random 
import math
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../py_aff3ct/build/lib')
import py_aff3ct as aff3ct

from py_aff3ct.module.py_module import Py_Module

from source           import source
from add_impulses     import add_impulses
from factory          import factory
from display_impulses import display_impulses

import argparse

def generate(code_type, N, K, EbN0, I):
        
    fac = factory(code_type, N, K, EbN0, I)

    enc, dec  = fac.build()
    sigma_val = fac.sigma_val

    src  = aff3ct.module.source.Source_random_fast(K, 12)
    mdm  = aff3ct.module.modem.Modem_BPSK_fast(N)

    gen = aff3ct.tools.Gaussian_noise_generator_implem.FAST
    chn = aff3ct.module.channel.Channel_AWGN_LLR(N, gen)
    mnt = aff3ct.module.monitor.Monitor_BFER_AR(K, 1)

    sigma      = np.ndarray(shape = (1,1),  dtype = np.float32)
    noisy_vec  = np.ndarray(shape = (1,N),  dtype = np.float32)
    vec_src    = np.ndarray(shape = (1,K),  dtype = np.int32)
    r_in       = np.ndarray(shape = (1,N),  dtype = np.float32)

    src.reset()

    seed = random.randint(0, 123456789)
    src.set_seed(seed+1)
    chn.set_seed(seed)

    src["generate   ::U_K  "] = enc["encode       ::U_K "]
    enc["encode     ::X_N  "] = mdm["modulate     ::X_N1"]
    mdm["modulate   ::X_N2 "] = chn["add_noise    ::X_N "]
    chn["add_noise  ::Y_N  "] = mdm["demodulate   ::Y_N1"]
    mdm["demodulate ::Y_N2 "] = dec["decode_siho  ::Y_N "]
    src["generate   ::U_K  "] = mnt["check_errors ::U   "]
    dec["decode_siho::V_K  "] = mnt["check_errors ::V   "]
    chn["add_noise  ::CP   "].bind( sigma  )
    mdm["demodulate ::CP   "].bind( sigma  )

    seq1 = aff3ct.tools.sequence.Sequence(src("generate"),  1)
    seq1.export_dot("chaos.dot")

    for i in range(len(sigma_val)):
        sigma[:] = sigma_val[i]
        seq1.exec()		
        print("be=",mnt.get_n_be())
        print("fe=",mnt.get_n_fe())
        mnt.reset()
    with open('noisy_cw.npy', 'wb') as f:
        np.save(f, dec['decode_siho::Y_N'][:])
        np.save(f, src['generate::U_K'][:])
