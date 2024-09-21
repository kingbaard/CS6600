#!/usr/bin/python

#################################################
# model: cs5600_6600_f24_hw02_data.py
# description: the data for the hw02 unit tests
# bugs to vladimir kulyukin in canvas
#################################################

import numpy as np

# Problem 1: AND, OR, NOT, XOR
X1 = np.array([[0, 0],
               [1, 0],
               [0, 1],
               [1, 1]])

X2 = np.array([[0],
               [1]])

# Training data for the boolean expression in HW02.
X3 = np.array([[1, 1, 1, 1], #1
               [0, 1, 1, 1], #2
               [1, 0, 1, 1], #3
               [1, 1, 0, 1], #4
               [1, 1, 1, 0], #5
               [0, 0, 1, 1], #6
               [0, 1, 0, 1], #7
               [0, 1, 1, 0], #8
               [1, 0, 0, 1], #9
               [1, 0, 1, 0], #10
               [1, 1, 0, 0], #11
               [1, 0, 0, 0], #12
               [0, 1, 0, 0], #13
               [0, 0, 1, 0], #14
               [0, 0, 0, 1], #15
               [0, 0, 0, 0]  #16
               ]
              )

# Ground Truth for AND-Function
y_and = np.array([[0],
                  [0],
                  [0],
                  [1]])

# Ground Truth for OR-Function
y_or = np.array([[0],
                 [1],
                 [1],
                 [1]])

# Ground Truth for XOR-Function
y_xor = np.array([[0],
                  [1],
                  [1],
                  [0]])

# Ground Truth for NOT-Function
y_not = np.array([[1],
                  [0]])

# Ground Truth for Boolean Expression
bool_exp = np.array([[1], #1
                     [0], #2
                     [0], #3
                     [1], #4
                     [1], #5
                     [0], #6
                     [0], #7
                     [0], #8
                     [0], #9
                     [0], #10
                     [1], #11
                     [1], #12
                     [1], #13
                     [0], #14
                     [0], #15
                     [1], #16
                     ]
                    )

