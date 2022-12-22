
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QListWidget,QListWidgetItem,QMessageBox
from pyqtgraph import PlotWidget

import sys
sys.path.insert(0, '../pyaf/py_aff3ct/build/lib')
#sys.path.insert(0, '/home/bekri/projetavanceS9/pyaf/py_aff3ct/build/lib')
import numpy as np
import py_aff3ct as aff3ct
import math
import time
import matplotlib.pyplot as plt
from datetime import timedelta
from tempfile import TemporaryFile

from py_aff3ct.module.py_module import Py_Module
import py_aff3ct.tools.frozenbits_generator as tool_fb
import py_aff3ct.tools.noise as tool_noise
import py_aff3ct.tools.sparse_matrix as tool_sp
import random 

from source           import source
from add_impulses     import add_impulses
from factory          import factory
from display_impulses import display_impulses


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")
        MainWindow.resize(548, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        #self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        #self.lineEdit.setObjectName("lineEdit")
        #self.lineEdit.setGeometry(130, 80, 113, 25)
        #self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        #self.lineEdit_2.setObjectName("lineEdit_2")
        #self.lineEdit_2.setGeometry(130, 120, 113, 25)
        #self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        #self.lineEdit_3.setObjectName("lineEdit_3")
        #self.lineEdit_3.setGeometry(130, 160, 113, 25)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.label.setGeometry(100, 40, 171, 20)
        #self.label_2 = QtWidgets.QLabel(self.centralwidget)
        #self.label_2.setObjectName("label_2")
        #self.label_2.setGeometry(20, 80, 101, 20)
        #self.label_3 = QtWidgets.QLabel(self.centralwidget)
        #self.label_3.setObjectName("label_3")
        #self.label_3.setGeometry(20, 120, 101, 20)
        #self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        #self.lineEdit_5.setObjectName("lineEdit_5")
        #self.lineEdit_5.setGeometry(130, 240, 113, 25)
        #self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        #self.lineEdit_6.setObjectName("lineEdit_6")
        #self.lineEdit_6.setGeometry(130, 200, 113, 25)
        #self.label_4 = QtWidgets.QLabel(self.centralwidget)
        #self.label_4.setObjectName("label_4")
        #self.label_4.setGeometry(30, 160, 91, 17)
        #self.label_5 = QtWidgets.QLabel(self.centralwidget)
        #self.label_5.setObjectName("label_5")
        #self.label_5.setGeometry(30, 200, 91, 17)
        #self.label_6 = QtWidgets.QLabel(self.centralwidget)
        #self.label_6.setObjectName("label_6")
        #self.label_6.setGeometry(30, 240, 91, 17)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setGeometry(200, 300, 141, 25)
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

        self.retranslateUi(MainWindow)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        #self.lineEdit.setText("")
        #self.label.setText(_translate("MainWindow", "PARAMETRES"))
        #self.label_2.setText(_translate("MainWindow", "VALEUR DE k"))
        #self.label_3.setText(_translate("MainWindow", "VALEUR DE n"))
        #self.lineEdit_6.setText("")
        #self.label_4.setText(_translate("MainWindow", "ebn0_min"))
        #self.label_5.setText(_translate("MainWindow", "ebn0_max"))
        #self.label_6.setText(_translate("MainWindow", "ebn0_step"))
        self.pushButton.setText(_translate("MainWindow", "générer l'image"))
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
            #K=25
            #N=50
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

        my_src = source(r_in, N, ix_x, ix_y)
        adi    = add_impulses(N)
        enc, dec2   = fac.build()
        mnt2   = aff3ct.module.monitor.Monitor_BFER(K, 1)
        dis    = display_impulses(H, W, heat_map, h_ix)

        my_src["generate :: r_in   "   ] = adi   ["add :: r_in   "    ] 
        my_src["generate :: enable "   ] = adi   ["add :: enable "    ]   
        my_src["generate :: delta_x"   ] = adi   ["add :: delta_x"    ]
        my_src["generate :: delta_y"   ] = adi   ["add :: delta_y"    ]
        my_src["generate :: ix_x   "   ] = adi   ["add :: ix_x   "    ]
        my_src["generate :: ix_y   "   ] = adi   ["add :: ix_y   "    ]
        adi   ["add           :: r_out"] = dec2  ["decode_siho  ::Y_N"]
        dec2  ["decode_siho   :: V_K  "] = mnt2  ["check_errors2::V"  ]
        dis   ["display :: x"          ] = my_src["generate :: x"     ]
        dis   ["display :: y"          ] = my_src["generate :: y"     ]
        dis   ["display :: BE "        ] = mnt2  ["check_errors2 :: BE"]
        dis   ["display :: enable"     ] = my_src["generate :: enable"] 

        mnt2  ["check_errors2:: U    " ].bind(vec_src)
        mnt2.create_reset_task()
        mnt2  ["reset" ] = dis["display::status"]	

        seq2 = aff3ct.tools.sequence.Sequence(my_src("generate"),  1)
        seq2.export_dot("full_module.dot")

        seq2.exec()
        plt.imshow(dis.map, interpolation='none')
        plt.show()
    
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())