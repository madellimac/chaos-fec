
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QListWidget,QListWidgetItem,QMessageBox,QLabel
from PyQt5.QtWidgets import QGridLayout,QMainWindow, QApplication, QLabel
from pyqtgraph import PlotWidget
from PyQt5.QtGui import QPixmap
import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
sys.path.insert(0, '../pyaf/build/lib')

import numpy as np
import py_aff3ct as aff3ct
import pyaf
import math
import time
import matplotlib.pyplot as plt
from   datetime import timedelta
from   tempfile import TemporaryFile
import os 

from   py_aff3ct.module.py_module import Py_Module
import py_aff3ct.tools.frozenbits_generator as tool_fb
import py_aff3ct.tools.noise as tool_noise
import py_aff3ct.tools.sparse_matrix as tool_sp
import random 

from factory          import factory
from display_impulses import display_impulses


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")
        MainWindow.resize(548, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.label.setGeometry(100, 40, 171, 20)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setGeometry(200, 300, 141, 25)
        
        self.checkBox_image = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_image.setObjectName("checkBox_image")
        self.checkBox_image.setGeometry(200, 350, 180, 23)
        
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.label_7.setGeometry(330, 40, 81, 17)
        
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setObjectName("checkBox")
        self.checkBox.setGeometry(330, 80, 92, 23)
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_2.setGeometry(330, 120, 92, 23)
        self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_3.setGeometry(330, 160, 92, 23)
        self.checkBox_4 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_4.setObjectName("checkBox_4")
        self.checkBox_4.setGeometry(330, 200, 92, 23)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(0, 0, 548, 22)
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        labelpic = QtWidgets.QLabel(self.centralwidget)
        
        self.retranslateUi(MainWindow)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        self.pushButton.setText(_translate("MainWindow", "générer l'image"))
        self.checkBox_image.setText(_translate("MainWindow", "sauvegarder l'image"))
        self.pushButton.clicked.connect(self.afficher_image)
        self.label_7.setText(_translate("MainWindow", "Code type"))
        self.checkBox.setText(_translate("MainWindow", "turbo"))
        self.checkBox_2.setText(_translate("MainWindow", "polar"))
        self.checkBox_3.setText(_translate("MainWindow", "ldpc"))
        self.checkBox_4.setText(_translate("MainWindow", "bch"))


    def afficher_image(self):       
        code_type = ""
        if(self.checkBox.isChecked()):  #turbo 
            code_type="turbo"
        elif(self.checkBox_2.isChecked()):
            code_type="polar"
        elif(self.checkBox_3.isChecked()):  #ldpc
            code_type="ldpc"
        elif(self.checkBox_4.isChecked()):
            code_type="bch"
        else : 
            print("Error : code type is not supported.")
        
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
        ebn0 = np.asarray([2.8])
        I = 8
        load_file = False

        delta_x_range = np.arange(-4,4,0.05)
        delta_y_range = np.arange(-4,4,0.05)

        W = len(delta_x_range)
        H = len(delta_y_range)

        fac = factory(code_type, N, K, ebn0, I)

        enc, dec = fac.build()
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

        map = np.zeros( (H,W,3), dtype=np.uint8)
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
        
        if(load_file == False):
            for i in range(len(sigma_val)):
                sigma[:] = sigma_val[i]
                seq1.exec()		
                print("be=",mnt.get_n_be())
                print("fe=",mnt.get_n_fe())
                mnt.reset()
            with open('noisy_cw.npy', 'wb') as f:
                np.save(f, dec['decode_siho::Y_N'][:])
                np.save(f, src['generate::U_K'][:])
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

        cdc         = pyaf.conductor.Conductor(noisy_vec[0,:], N, ix_x, ix_y) # Conductor
        adi         = pyaf.add.Add_impulses(N)
        enc, dec2   = fac.build()
        mnt2   		= aff3ct.module.monitor.Monitor_BFER(K, 1)
        dis    		= display_impulses(H, W, heat_map, h_ix)


        cdc   ["generate :: noisy_vec" ] = adi   ["add :: r_in   "     ] 
        cdc   ["generate :: delta_x"   ] = adi   ["add :: delta_x"     ]
        cdc   ["generate :: delta_y"   ] = adi   ["add :: delta_y"     ]
        cdc   ["generate :: ix_x   "   ] = adi   ["add :: ix_x   "     ]
        cdc   ["generate :: ix_y   "   ] = adi   ["add :: ix_y   "     ]
        adi   ["add           :: r_out"] = dec2  ["decode_siho  ::Y_N" ]
        dec2  ["decode_siho   :: V_K  "] = mnt2  ["check_errors2::V"   ]
        dis   ["display :: x"          ] = cdc   ["generate :: x"      ]
        dis   ["display :: y"          ] = cdc   ["generate :: y"      ]
        dis   ["display :: BE "        ] = mnt2  ["check_errors2 :: BE"]

        mnt2  ["check_errors2:: U"].bind(vec_src)
        mnt2.create_reset_task()
        mnt2  ["reset" ] = dis["display::status"]
        seq2 = aff3ct.tools.sequence.Sequence(cdc("generate"),  1)
        seq2.export_dot("full_module.dot")

        seq2.exec()
        plt.imshow(dis.map, interpolation='none')
        
        if (self.checkBox_image.isChecked()): 
            plt.savefig('chaosfig.png')
            print("saved image")
            
        plt.show()
        print("affichage réussie!")
    
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
