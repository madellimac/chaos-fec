import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
import numpy as np
import py_aff3ct as aff3ct
import math
import py_aff3ct.tools.frozenbits_generator as tool_fb
import py_aff3ct.tools.noise as tool_noise
import py_aff3ct.tools.sparse_matrix as tool_sp

class factory:

    def build(self):

        if(self.code_type == "turbo"):
            
            esn0 = self.ebn0 + 10 * math.log10(self.K/self.N)
            self.sigma_val = 1/(math.sqrt(2) * 10 ** (esn0 / 20))

            # Build DVBS-RCS2 Turbo encoder.
            enc_n = aff3ct.module.encoder.Encoder_RSC_DB(self.K,2*self.K,standard='DVB-RCS1')
            enc_i = aff3ct.module.encoder.Encoder_RSC_DB(self.K,2*self.K,standard='DVB-RCS1')

            # Build DVBS-RCS2 Interleaver.
            itl_core = aff3ct.tools.interleaver_core.Interleaver_core_ARP_DVB_RCS1(self.K//2)
            itl_bit  = aff3ct.module.interleaver.Interleaver_int32(itl_core)
            itl_llr  = aff3ct.module.interleaver.Interleaver_float(itl_core)
            enc      = aff3ct.module.encoder.Encoder_turbo_DB(self.K, self.N, enc_n, enc_i, itl_bit)

            # Build DVBS-RCS2 Turbo decoder.
            trellis_n = enc_n.get_trellis()
            trellis_i = enc_i.get_trellis()

            dec_n = aff3ct.module.decoder.Decoder_RSC_DB_BCJR_DVB_RCS1(self.K,trellis_n)
            dec_i = aff3ct.module.decoder.Decoder_RSC_DB_BCJR_DVB_RCS1(self.K,trellis_i)
            dec   = aff3ct.module.decoder.Decoder_turbo_DB(self.K, self.N, self.I, dec_n, dec_i, itl_llr)

        elif(self.code_type == "polar"): #polar

            esn0 = self.ebn0 + 10 * math.log10(self.K/self.N)
            self.sigma_val = 1/(math.sqrt(2) * 10 ** (esn0 / 20))
        
            fbgen = tool_fb.Frozenbits_generator_GA_Arikan(self.K, self.N)
            noise = tool_noise.Sigma(self.sigma_val)
            fbgen.set_noise(noise)
            frozen_bits = fbgen.generate()

            enc  = aff3ct.module.encoder.Encoder_polar_sys      (self.K,self.N,frozen_bits)     # Build the encoder
            #dec  = aff3ct.module.decoder.Decoder_polar_SC_fast_sys(K,N,frozen_bits)   # Build the decoder
            #dec = aff3ct.module.decoder.Decoder_polar_SCL_MEM_fast_sys(K,N,8,frozen_bits)
            dec = aff3ct.module.decoder.Decoder_polar_SCAN_naive_sys(self.K, self.N, self.I, frozen_bits)

        elif(self.code_type == "ldpc"):
            esn0 = self.ebn0 + 10 * math.log10(K/N)
            self.sigma_val = 1/(math.sqrt(2) * 10 ** (esn0 / 20))

            H   = tool_sp.alist.read("../py_aff3ct/lib/aff3ct/conf/dec/LDPC/WIMAX_288_576.alist")
            # H   = tool_sp.alist.read("../py_aff3ct/lib/aff3ct/conf/dec/LDPC/10GBPS-ETHERNET_1723_2048.alist")
            #H = tool_sp.alist.read("../py_aff3ct/lib/aff3ct/conf/dec/LDPC/CCSDS_64_128.alist")
            
            N   = H.shape[0]
            m   = H.shape[1]
            K   = N - m
            enc  = aff3ct.module.encoder.Encoder_LDPC_from_H    (K, N, H)                                                   # Build the encoder
            dec  = aff3ct.module.decoder.Decoder_LDPC_BP_horizontal_layered_inter_NMS (K, N, self.I, H, enc.get_info_bits_pos()) # Build the decoder
        elif(self.code_type == "bch"):
            t = 3
            poly = aff3ct.tools.BCH_polynomial_generator(N, t)
            enc = aff3ct.module.encoder.Encoder_BCH(K, N, poly)
            dec = aff3ct.module.decoder.Decoder_BCH_std(K, N, poly)
        else:
            print("Error : Non supported code type, must be : polar/ldpc/turbo/bch")
            #exit()

            
        return enc, dec
    
    def __init__(self, code_type, N, K, ebn0, I):

        self.code_type = code_type
        self.N = N   
        self.K = K
        self.ebn0 = ebn0
        self.I = I

        print("init factory")