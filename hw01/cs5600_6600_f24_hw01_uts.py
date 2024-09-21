
####################################################
# CS 5600/6600: F24: Assignment 1: Unit Tests
# author: vladimir kulyukin
# bugs to vladimir kulyukin in canvas
# I ran these UTs in Python 3.10.12 on Ubuntu 22.04.4
#####################################################

import numpy as np
from  cs5600_6600_f24_hw01 import *
import unittest

class cs5600_6600_f24_hw01_uts(unittest.TestCase):

    # let's test the and-percep.
    def test_assgn_01_ut_01(self):
        print('\n***** Assign 01: UT 01 ************')
        x00 = np.array([0, 0])
        x01 = np.array([0, 1])
        x10 = np.array([1, 0])
        x11 = np.array([1, 1])

        andp = and_percep()
        assert andp.output(x00) == 0
        assert andp.output(x01) == 0
        assert andp.output(x10) == 0
        assert andp.output(x11) == 1
        print('Assign 01: UT 01: passed...')

    # let's test the or-percep.
    def test_assgn_01_ut_02(self):
        print('\n***** Assign 01: UT 02 ************')
        x00 = np.array([0, 0])
        x01 = np.array([0, 1])
        x10 = np.array([1, 0])
        x11 = np.array([1, 1])
        orp = or_percep()
        assert orp.output(x00) == 0
        assert orp.output(x01) == 1
        assert orp.output(x10) == 1
        assert orp.output(x11) == 1
        print('Assign 01: UT 02: passed...')


    # let's test the not-percep.
    def test_assgn_01_ut_03(self):
        print('\n***** Assign 01: UT 03 ************')
        notp = not_percep()
        assert notp.output(np.array([0])) == 1
        assert notp.output(np.array([1])) == 0
        print('Assign 01: UT 03: passed...')

    
    # let's test the 1st xor-percep.
    def test_assgn_01_ut_04(self):
        print('\n***** Assign 01: UT 04 ************')
        x00 = np.array([0, 0])
        x01 = np.array([0, 1])
        x10 = np.array([1, 0])
        x11 = np.array([1, 1])
        xorp = xor_percep()
        assert xorp.output(x00) == 0
        assert xorp.output(x01) == 1
        assert xorp.output(x10) == 1
        assert xorp.output(x11) == 0
        print('Assign 01: Unit Test 04: passed...')
    
    # # let's test the 2nd xor-percep.
    def test_assgn_01_ut_05(self):
        print('\n***** Assign 01: UT 05 ************')
        x00 = np.array([0, 0])
        x01 = np.array([0, 1])
        x10 = np.array([1, 0])
        x11 = np.array([1, 1])
        xorp2 = xor_percep2()
        assert xorp2.output(x00)[0] == 0
        assert xorp2.output(x01)[0] == 1
        assert xorp2.output(x10)[0] == 1
        assert xorp2.output(x11)[0] == 0
        print('Assign 01: UT 05: passed...')

    # let's test perceptron_network.
    def test_hw01_ut06(self):
        print('\n***** Assign 01: UT 06 ************')
        pn = percep_net()
        x0000 = np.array([0, 0, 0, 0])
        x0100 = np.array([0, 1, 0, 0])
        x1100 = np.array([1, 1, 0, 0])
        x1101 = np.array([1, 1, 0, 1])
        x1110 = np.array([1, 1, 1, 0])                
        assert pn.output(x0000) == 0
        assert pn.output(x0100) == 1
        assert pn.output(x1100) == 1
        assert pn.output(x1101) == 1
        assert pn.output(x1110) == 0        
        print('Assign 01: UT 06 passed...')        

### ================ Unit Tests ====================

if __name__ == '__main__':
    unittest.main()


 
    
        
        

    
        





