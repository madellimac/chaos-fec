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

def generate(code_type, N, K, EbN0, I):

	delta_x_range = np.arange(-4, 4, 0.05)
	delta_y_range = np.arange(-4, 4, 0.05)

	W = len(delta_x_range)
	H = len(delta_y_range)

	fac = factory(code_type, N, K, EbN0, I)

	sigma      = np.ndarray(shape = (1,1),  dtype = np.float32)
	noisy_vec  = np.ndarray(shape = (1,N),  dtype = np.float32)
	vec_src    = np.ndarray(shape = (1,K),  dtype = np.int32)
	r_in       = np.ndarray(shape = (1,N),  dtype = np.float32)

	map = np.zeros( (H,W,3), dtype=np.uint8)

	with open('noisy_cw.npy', 'rb') as f:
		r_in = np.load(f)
		vec_src = np.load(f)

	# Pour le module source
	ix_x    = np.ndarray(shape = (1,1),  dtype = np.int32  )
	ix_y    = np.ndarray(shape = (1,1),  dtype = np.int32  )
	ix_x[:] = 1  # np.random.randint(N)
	ix_y[:] = 20 # (ix_x[:]+5) % N 

	# Pour le module display
	heat_map = np.ndarray(shape=(5,3),dtype = np.uint8)
	heat_map[0] = [4,8,16]
	heat_map[1] = [16,8,4]
	heat_map[2] = [8,16,4]
	heat_map[3] = [100,50,25]
	heat_map[4] = [16,16,16]
	h_ix = 0

	my_src = source(r_in, N, ix_x, ix_y)
	adi    = add_impulses(N)
	enc, dec2   = fac.build()
	
	mnt2   = aff3ct.module.monitor.Monitor_BFER(K, 1)
	dis    = display_impulses(H, W, heat_map, h_ix)

	my_src["generate :: r_in   "   ] = adi   ["add :: r_in   "     ] 
	my_src["generate :: enable "   ] = adi   ["add :: enable "     ]   
	my_src["generate :: delta_x"   ] = adi   ["add :: delta_x"     ]
	my_src["generate :: delta_y"   ] = adi   ["add :: delta_y"     ]
	my_src["generate :: ix_x   "   ] = adi   ["add :: ix_x   "     ]
	my_src["generate :: ix_y   "   ] = adi   ["add :: ix_y   "     ]
	adi   ["add           :: r_out"] = dec2  ["decode_siho  ::Y_N" ]
	dec2  ["decode_siho   :: V_K  "] = mnt2  ["check_errors2::V"   ]
	dis   ["display :: x"          ] = my_src["generate :: x"      ]
	dis   ["display :: y"          ] = my_src["generate :: y"      ]
	dis   ["display :: BE "        ] = mnt2  ["check_errors2 :: BE"]
	dis   ["display :: enable"     ] = my_src["generate :: enable" ] 

	mnt2  ["check_errors2:: U    " ].bind(vec_src)
	mnt2.create_reset_task()
	mnt2  ["reset" ] = dis["display::status"]

	seq2 = aff3ct.tools.sequence.Sequence(my_src("generate"),  1)
	seq2.export_dot("full_module.dot")

	seq2.exec()
	plt.imshow(dis.map, interpolation='none')
	plt.show()

	## Il semblerait qu'il y ai un offset de -1 selon x