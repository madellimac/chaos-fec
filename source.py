import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
import numpy as np
import py_aff3ct as aff3ct
from py_aff3ct.module.py_module import Py_Module

class source(Py_Module):

	def generate(self, r_in, enable, delta_x, delta_y, ix_x, ix_y, x, y):
		r_in[:] 	 = self.r_in[:]
		enable[:]	 = self.enable
		ix_x[:] 	 = self.ix_x[:]
		ix_y[:] 	 = self.ix_y[:]
		x[:] 		 = self.vec_cnt // len(self.delta_y_range)
		y[:] 		 = self.vec_cnt %  len(self.delta_y_range)
		delta_x[:] 	 = self.delta_x_range[x]	
		delta_y[:] 	 = self.delta_y_range[y]	
		self.vec_cnt = self.vec_cnt + 1 	
		return 0
    
	def __init__(self, r_in, N, ix_x, ix_y): # N == len(r_in[0,:])

		Py_Module.__init__(self) # Call the aff3ct Py_Module __init__
		
		self.name   		= "source_xy"   
		self.r_in   		= r_in				
		self.delta_x_range 	= np.arange(-4,4,0.05)
		self.delta_y_range 	= np.arange(-4,4,0.05)
		self.enable 		= 1
		self.ix_x   		= ix_x 
		self.ix_y   		= ix_y 
		self.x				= 0			
		self.y				= 0						
		self.vec_cnt 		= 0 			
		self.delta_x 		= np.ndarray(shape = (1,1), dtype = np.float32)
		self.delta_y		= np.ndarray(shape = (1,1), dtype = np.float32)

		t_generate = self.create_task("generate") 	# create a task for your module

		s_r_in    = self.create_socket_out  (t_generate, "r_in"   , N		 , np.float32 ) # create an output socket for the task t_generate
		s_enable  = self.create_socket_out  (t_generate, "enable" , 1		 , np.int32   ) # create an output socket for the task t_generate
		s_delta_x = self.create_socket_out  (t_generate, "delta_x", 1		 , np.float32 )	# create an output socket for the task t_generate
		s_delta_y = self.create_socket_out  (t_generate, "delta_y", 1	 	 , np.float32 ) # create an output socket for the task t_generate
		s_ix_x    = self.create_socket_out  (t_generate, "ix_x"   , 1		 , np.int32   ) # create an output socket for the task t_generate
		s_ix_y    = self.create_socket_out  (t_generate, "ix_y"   , 1		 , np.int32   ) # create an output socket for the task t_generate
		s_x    	  = self.create_socket_out  (t_generate, "x"      , 1		 , np.int32   ) # create an output socket for the task t_generate
		s_y    	  = self.create_socket_out  (t_generate, "y"   	  , 1		 , np.int32   ) # create an output socket for the task t_generate

		self.create_codelet(t_generate, lambda slf, lsk, fid: slf.generate(lsk[s_r_in],lsk[s_enable],lsk[s_delta_x],lsk[s_delta_y],lsk[s_ix_x],lsk[s_ix_y],lsk[s_x], lsk[s_y]))