import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
import numpy as np
import py_aff3ct as aff3ct
from py_aff3ct.module.py_module import Py_Module

class add_impulses(Py_Module):

	def add(self, ix_x, ix_y, delta_x, delta_y, enable, r_in, r_out):
		r_out[0,:] = r_in[:]
		if(enable == 1):
			r_out[0, ix_x] = delta_x
			r_out[0, ix_y] = delta_y
		return 0
    
	def __init__(self, N):

		Py_Module.__init__(self) # Call the aff3ct Py_Module __init__
		self.name = "add_impulses"   # Set your module's name

		t_add = self.create_task("add") # create a task for your module
        
		s_ix_x    = self.create_socket_in  (t_add, "ix_x"   , 1, np.int32   ) # create an input socket for the task t_add
		s_ix_y    = self.create_socket_in  (t_add, "ix_y"   , 1, np.int32   ) # create an input socket for the task t_add
		s_delta_x = self.create_socket_in  (t_add, "delta_x", 1, np.float32 ) # create an input socket for the task t_add
		s_delta_y = self.create_socket_in  (t_add, "delta_y", 1, np.float32 ) # create an input socket for the task t_add
		s_enable  = self.create_socket_in  (t_add, "enable" , 1, np.int32   ) # create an input socket for the task t_add
		s_r_in    = self.create_socket_in  (t_add, "r_in"   , N, np.float32 ) # create an input socket for the task t_add
		s_r_out   = self.create_socket_out (t_add, "r_out"  , N, np.float32 ) # create an output socket for the task t_add
		    
		self.create_codelet(t_add, lambda slf, lsk, fid: slf.add(lsk[s_ix_x], lsk[s_ix_y], lsk[s_delta_x], lsk[s_delta_y], lsk[s_enable], lsk[s_r_in], lsk[s_r_out])) # create codelet