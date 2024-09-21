#/usr/bin/python

############################################
# module: cs5600_6600_f24_hw03.py
# starter code for CS5600/6600: F24: HW03
# bugs to vladimir kulyukin in canvas.
############################################

'''
Problem 3: 1-layer NN
------------------------
Eta/HLS |   10    |    25   |   50    |
------------------------
0.5     |  89.08  |  92.29  |  93.25  |
------------------------
0.25    |  88.22  |  91.06  |  83.03  |
------------------------
0.125   |  87.36  |  88.55  |  80.55  |
------------------------

Problem 3: two-hidden layer NNs
h1 = 10
------------------------
Eta/h2  |   10    |   25    |    50   |
------------------------
0.5     |  88.91  |  90.46  |  90.84  |
------------------------
0.25    |  87.91  |  88.41  |  89.51  |
------------------------
0.125   |  85.06  |  88.29  |  86.92  |
------------------------

h1 = 25
------------------------
Eta/h2  |   10    |   25    |    50   |
------------------------
0.5     |  91.97  |  91.96  |  92.86  |
------------------------
0.25    |  90.73  |  91.16  |  91.54  |
------------------------
0.125   |  87.24  |  89.11  |  90.38  |
------------------------

h1 = 50
------------------------
Eta/h2  |   10    |   25    |    50   |
------------------------
0.5     |  92.08  |  93.22  |  93.68  |
------------------------
0.25    |  91.51  |  90.76  |  91.82  |
------------------------
0.125   |  88.49  |  89.35  |  81.16  |
------------------------
'''

from ann import ann
from mnist_loader import load_data_wrapper

train_d, valid_d, test_d = load_data_wrapper()

HLS = [10, 25, 50]
ETA = [0.5, 0.25, 0.125]

def train_1_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10):
    for hls_value in hls:
        size = [784, hls_value, 10]
        for eta_value in eta:
            print(f"*** Training 784x{hls_value}x10 ANN with eta={eta_value} ***")
            nn = ann(size)
            nn.mini_batch_sgd(train_d, num_epochs, mini_batch_size, eta_value, test_d)

def train_2_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10):
    for h1 in hls:
        for h2 in hls:
            size = [784, h1, h2, 10]
            for eta_value in eta:
                print(f"*** Training 784x{h1}x{h2}x10 ANN with eta={eta_value} ***")
                nn = ann(size)
                nn.mini_batch_sgd(train_d, num_epochs, mini_batch_size, eta_value, test_d)

### Uncomment to run
if __name__ == '__main__':
    # train_1_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10)
    train_2_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10)
