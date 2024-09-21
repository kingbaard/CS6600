
#####################################################
# CS 5600/6600: F24: Assignment 2: Unit Tests
# bugs to vladimir kulyukin in canvas
#####################################################

import numpy as np
from  cs5600_6600_f24_hw02 import *
import pickle
import unittest

class cs5600_6600_f24_hw02_uts(unittest.TestCase):

   
    def test_assgn_02_ut_01(self):
        print('\nUT 01')
        wmats = build_nn_wmats((2, 3, 1))
        print(wmats[0])
        assert wmats[0].shape == (2, 3)
        print(wmats[1])
        assert wmats[1].shape == (3, 1)

    def test_assgn_02_ut_02(self):
        print('\nUT 02')
        wmats = build_nn_wmats((8, 3, 8))
        # print(wmats[0])
        assert wmats[0].shape == (8, 3)
        # print(wmats[1])
        assert wmats[1].shape == (3, 8)

    def test_assgn_02_ut_03(self, thresh=0.4):
        print('\nUT 03')
        num_iters = 5000
        and_wmats = train_3_layer_nn(num_iters, X1, y_and, build_231_nn)
        print('\nTraining & Testing 2x3x1 AND ANN Thresholded at {} for {} iters'.format(thresh, num_iters))
        for i in range(len(X1)):
            print('{}, {} --> {}'.format(X1[i], fit_3_layer_nn(X1[i], and_wmats), y_and[i]))
            assert (fit_3_layer_nn(X1[i], and_wmats, thresh=0.4, thresh_flag=True) == y_and[i]).all()
        print('\n')

    def test_assgn_02_ut_04(self, thresh=0.4):
        print('\nUT 04')
        num_iters = 500
        and_wmats = train_4_layer_nn(num_iters, X1, y_and, build_2331_nn)
        print('\nTraining & Testing 2x3x3x1 AND ANN Thresholded at {} for {} iters'.format(thresh, num_iters))
        for i in range(len(X1)):
            print('{}, {} --> {}'.format(X1[i], fit_4_layer_nn(X1[i], and_wmats), y_and[i]))
            assert (fit_4_layer_nn(X1[i], and_wmats, thresh=0.4, thresh_flag=True) == y_and[i]).all()
        print('\n')

    def test_assgn_02_ut_05(self, thresh=0.4):
        print('\nUT 05')
        num_iters = 500
        or_wmats = train_3_layer_nn(num_iters, X1, y_or, build_231_nn)
        print('\nTraining & Testing 2x3x1 OR ANN Thresholded at {} for {} iters'.format(thresh, num_iters))    
        for i in range(len(X1)):
            print('{}, {} --> {}'.format(X1[i], fit_3_layer_nn(X1[i], or_wmats), y_or[i]))
            assert (fit_3_layer_nn(X1[i], or_wmats, thresh=0.4, thresh_flag=True) == y_or[i]).all()
        print('\n')

    def test_assgn_02_ut_06(self, thresh=0.4):
        print('\nUT 06')
        num_iters = 700        
        xor_wmats = train_4_layer_nn(num_iters, X1, y_xor, build_2331_nn)
        print('\nTraining & Testing 2x3x3x1 XOR ANN Thresholded at {} for {} iters'.format(thresh, num_iters))    
        for i in range(len(X1)):
            print('{}, {} --> {}'.format(X1[i], fit_4_layer_nn(X1[i], xor_wmats), y_xor[i]))
            assert (fit_4_layer_nn(X1[i], xor_wmats, thresh=0.4, thresh_flag=True) == y_xor[i]).all()
        print('\n')

    def test_assgn_02_ut_07(self, thresh=0.4):
        print('\nUT 07')        
        num_iters = 500        
        not_wmats = train_3_layer_nn(num_iters, X2, y_not, build_121_nn)
        print('\nTraining & Testing 1x2x1 NOT ANN Thresholded at {} for {} iters'.format(thresh, num_iters))        
        for i in range(len(X2)):
            print('{}, {} --> {}'.format(X2[i], fit_3_layer_nn(X2[i], not_wmats), y_not[i]))
            assert (fit_3_layer_nn(X2[i], not_wmats, thresh=thresh, thresh_flag=True) == y_not[i]).all()
        print('\n')

    def test_assgn_02_ut_08(self, thresh=0.3):
        print('\nUT 08')
        num_iters = 800       
        bool_wmats = train_3_layer_nn(num_iters, X3, bool_exp, build_421_nn)
        print('\nTraining & Testing 4x2x1 BOOL EXP ANN Thresholded at {} for {} iters'.format(thresh, num_iters))        
        for i in range(len(X3)):
            print('{}, {} --> {}'.format(X3[i], fit_3_layer_nn(X3[i], bool_wmats), bool_exp[i]))
            assert (fit_3_layer_nn(X3[i], bool_wmats, thresh=thresh, thresh_flag=True) == bool_exp[i]).all()
        print('\n')

    def test_assgn_02_ut_09(self, thresh=0.3):
        print('\nUT 09')
        num_iters = 500        
        bool_wmats = train_4_layer_nn(num_iters, X3, bool_exp, build_4221_nn)
        print('\nTraining & Testing 4x2x2x1 BOOL EXP ANN Thresholded at {} for {} iters'.format(thresh, num_iters))        
        for i in range(len(X3)):
            print('{}, {} --> {}'.format(X3[i], fit_4_layer_nn(X3[i], bool_wmats), bool_exp[i]))
            assert (fit_4_layer_nn(X3[i], bool_wmats, thresh=0.3, thresh_flag=True) == bool_exp[i]).all()
        print('\n')
            
### ================ Unit Tests ====================

if __name__ == '__main__':
    unittest.main()


 
    
        
        

    
        





