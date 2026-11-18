import numpy as np
import matplotlib.pyplot as plt
from models.functions import ReLU, Linear, Sigmoid, SigmoidLayer, GELU
from utils import load_spiral, load_sst2

"""
** MLP Class is already implemented **
** Adjust Hyperparameters and experiment **
"""
np.random.seed(42)

class MLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size=1):
        
        # =============== EDIT HERE: START ===============
        # please do not change other parts in this code except EDIT HERE

        # ReLU, Sigmoid, GELU
        self.activation1 = Sigmoid() 
        self.activation2 = Sigmoid() 


        # =============== EDIT HERE: END =================


        self.input_layer = Linear(input_size, hidden_size1)
        self.hidden_layer = Linear(hidden_size1, hidden_size2)
        self.output_layer = SigmoidLayer(hidden_size2, output_size)

    def predict(self, x):
        x = self.input_layer.forward(x)
        x = self.activation1.forward(x)
        x = self.hidden_layer.forward(x)
        x = self.activation2.forward(x)
        pred = self.output_layer.predict(x)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

    def loss(self, x, y):
        x = self.input_layer.forward(x)
        x = self.activation1.forward(x)
        x = self.hidden_layer.forward(x)
        x = self.activation2.forward(x)
        loss = self.output_layer.forward(x, y)
        return loss

    def gradient(self):
        d_prev = 1
        d_prev = self.output_layer.backward(d_prev=d_prev)
        d_prev = self.activation2.backward(d_prev=d_prev)
        d_prev = self.hidden_layer.backward(d_prev=d_prev)
        d_prev = self.activation1.backward(d_prev=d_prev)
        self.input_layer.backward(d_prev)

    def update(self, learning_rate):
        self.input_layer.W -= self.input_layer.dW * learning_rate
        self.input_layer.b -= self.input_layer.db * learning_rate
        self.hidden_layer.W -= self.hidden_layer.dW * learning_rate
        self.hidden_layer.b -= self.hidden_layer.db * learning_rate
        self.output_layer.linear.W -= self.output_layer.linear.dW * learning_rate
        self.output_layer.linear.b -= self.output_layer.linear.db * learning_rate


# =============== EDIT HERE ===============
hidden_dim1 = 32
hidden_dim2 = 8
num_epochs = 200 # default: 100
batch_size = 16 # default: 16
learning_rate = 0.01   # default: 0.01

# =========================================

print_every = 10

learning_rate_decay_rate = 0.99
x_train, x_test, y_train, y_test= load_sst2('./data')
# x_train, y_train, x_test, y_test = load_spiral('./data')

num_feature = x_train.shape[1]
model = MLP(input_size=num_feature, hidden_size1=hidden_dim1, hidden_size2=hidden_dim2 , output_size=1)

num_data = len(x_train)
num_batch = int(np.ceil(num_data / batch_size))
for i in range(1, num_epochs + 1):
    epoch_loss = 0.0
    for b in range(0, len(x_train), batch_size):
        x_batch = x_train[b: b + batch_size]
        y_batch = y_train[b: b + batch_size]

        loss = model.loss(x_batch, y_batch)
        epoch_loss += loss

        model.gradient()
        model.update(learning_rate)
    epoch_loss /= num_batch

    # Train accuracy
    pred = model.predict(x_train)
    pred = pred.reshape(-1)

    total = len(x_train)
    correct = len(np.where(pred == y_train)[0])
    tr_acc = correct / total

    # Test accuracy
    pred = model.predict(x_test)
    pred = pred.reshape(-1)
    total = len(x_test)
    correct = len(np.where(pred == y_test)[0])
    te_acc = correct / total

    if i % print_every == 0:
        print('Train Accuracy = %.3f' % tr_acc)
        print('Test Accuracy = %.3f' % te_acc)
    learning_rate *= learning_rate_decay_rate
