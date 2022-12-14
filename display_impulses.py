import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
import numpy as np
import py_aff3ct as aff3ct
from py_aff3ct.module.py_module import Py_Module

class display_impulses(Py_Module):

	def display(self, x, y, BE, enable):#heat_map et h_ix sont des constantes
		if enable!=0:
			#self.map[self.H-self.ytempo-1,self.xtempo] = np.array([BE*self.heat_map[self.h_ix,0], BE*self.heat_map[self.h_ix,1], BE*self.heat_map[self.h_ix,2]])
			
			self.map[self.H-y-1,x,0] = BE*self.heat_map[self.h_ix,0]
			self.map[self.H-y-1,x,1] = BE*self.heat_map[self.h_ix,1]
			self.map[self.H-y-1,x,2] = BE*self.heat_map[self.h_ix,2]
			
			
			if self.xtempo==0:
				self.xtempo = 159
				if self.ytempo <159:
					self.ytempo+=1
				else:
					self.toggle_done()
			else:
				self.xtempo-=1
			
		return 0
    
	def __init__(self, H, W, heat_map, h_ix):
		
		Py_Module.__init__(self) # Call the aff3ct Py_Module __init__

		self.name 		= "display_impulses"   # Set your module's name
		self.xtempo 	= 0
		self.ytempo 	= 0
		self.heat_map = heat_map
		self.h_ix	 	= h_ix
		self.H 			= H
		self.W 			= W
		self.map		= np.zeros( (H,W,3), dtype=np.uint8)

		t_dis = self.create_task("display") # create a task for your module

		s_x 	 = self.create_socket_in  (t_dis, "x"   	, 1, np.int32   ) # create an input socket for the task t_dis
		s_y		 = self.create_socket_in  (t_dis, "y"   	, 1, np.int32   ) # create an input socket for the task t_dis
		s_BE	 = self.create_socket_in  (t_dis, "BE"   	, 1, np.int32   ) # create an input socket for the task t_dis
		s_enable = self.create_socket_in (t_dis, "enable"  , 1, np.int32   ) # create an input socket for the task t_dis
		
		self.create_codelet(t_dis, lambda slf, lsk, fid: slf.display(lsk[s_x], lsk[s_y], lsk[s_BE], lsk[s_enable])) # create codelet
