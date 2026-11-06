import numpy as np
import Decoding_failure
import Impulse_map
import Shuffle_noise
import random
import matplotlib.pyplot as plt

code_type = "ldpc"
if(code_type == "turbo"):
	K = 64*2
	N = 3*K
elif(code_type == "ldpc"):
	N = 576 
	K = 288
elif(code_type == "polar"):
	N = 128
	K = 64
elif(code_type == "bch"):
	N = 127
	K = 106
else:
	print("Error : code type is not supported.")
	exit()
	
EbN0 = np.asarray([2.8])
I = 2
T = N

Decoding_failure.generate(code_type, N, K, EbN0, I)

#Impulse_map.generate(code_type, N, K, EbN0, I)

error_array = np.ndarray(shape = (T,N),  dtype = np.int32)

cor = 0
error_sum = np.zeros(N,  dtype = np.int32)

for shift in range(0, T):
	error_array[shift] = Shuffle_noise.generate(code_type, N, K, EbN0, I, shift)
	be_nb = np.sum(error_array[shift])
	error_sum += error_array[shift]
	if(be_nb == 0):
		cor += 1
	#if(error_array[shift][0] == 0):
	#print(be_nb)
print(cor)
print(error_sum/T)
#error_map = np.zeros( (N,N), dtype=np.uint8)

print(error_array)
plt.figure
plt.imshow(255*error_array, interpolation='none')
plt.show()
plt.plot(error_sum/T)

