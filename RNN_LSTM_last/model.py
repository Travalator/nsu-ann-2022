import numpy as np
import matplotlib.pyplot as plt



from layers import FullyConnectedLayer,  RNN, LSTM




class RNN_Model:
    def __init__(self, features_num = 7, hidden_rnn_size = 20):
        self.layers = []
        self.name = 'rnn'
        self.layers.append(RNN(input_size=features_num, hidden_size=hidden_rnn_size))
        self.layers.append(FullyConnectedLayer(hidden_rnn_size, 1))
    


    def forward(self, X):
        
        
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)



    def predict(self, testx, testy, test_size, norm_coeff, plot = True):
        print(norm_coeff, 'norm')
        pred = self.forward(testx) * norm_coeff
        error = np.mean((pred - (testy * norm_coeff))**2)
        print(error)
        if plot:
            plt.figure(figsize=(30, 10))
            plt.plot(pred[:test_size])
            plt.plot(norm_coeff * testy[:test_size])


    def params(self):
        result = {}
        for layer_num, layer in enumerate(self.layers):
            for param_name, param in layer.params().items():
                result[f'{param_name} {layer.name}_{layer_num}'] = param

        return result
    
    def load_params(self, folder):
        for param_name, param in self.params().items():
            param.value = np.load(f'{folder}/{param_name}.npy')


class LSTM_Model:
    def __init__(self, features_num = 12, hidden_rnn_size = 10):
        self.layers = []
        self.name = 'lstm'
        self.layers.append(LSTM(input_size=features_num, hidden_size=hidden_rnn_size))
        self.layers.append(FullyConnectedLayer(hidden_rnn_size, 1))



    def forward(self, X):


        for layer in self.layers:
            X = layer.forward(X)
            # print(X.shape)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)





    def predict(self, testx, testy, test_size, norm_coeff, plot = True):
        print(norm_coeff, 'norm')
        pred = self.forward(testx) * norm_coeff
        error = np.mean((pred - (testy * norm_coeff))**2)
        print(error)
        if plot:
            plt.figure(figsize=(30, 10))
            plt.plot(pred[:test_size])
            plt.plot(norm_coeff * testy[:test_size])

    def params(self):
        result = {}
        for layer_num, layer in enumerate(self.layers):
            for param_name, param in layer.params().items():
                result[f'{param_name} {layer.name}_{layer_num}'] = param

        return result

    def load_params(self, folder):
        for param_name, param in self.params().items():
            param.value = np.load(f'{folder}/{param_name}.npy')

