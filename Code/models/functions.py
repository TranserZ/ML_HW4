import numpy as np

class ReLU:

    """
    ReLU Function. ReLU(x) = max(0, x)
    Implement forward & backward path of ReLU.
    """
    def __init__(self):
        self.zero_mask = None
    def forward(self, z):
        """
        ReLU Forward.
        ReLU(x) = max(0, x)

        z --> (ReLU) --> out

        [Inputs]
            z : ReLU input in any shape.

        [Outputs]
            out : ReLU(z).
        """
        out = None

        self.zero_mask = z < 0
        out = z.copy()
        out[self.zero_mask] = 0
        
        return out

    def backward(self, d_prev):
        """
        ReLU Backward.

        z --> (ReLU) --> out
        dz <-- (dReLU) <-- d_prev(dL/dout)

        [Inputs]
            d_prev : Gradients until now.
            d_prev = dL/dk, where k = ReLU(z).

        [Outputs]
            dz : Gradients w.r.t. ReLU input z.
        """
        dz = None
        # =============== EDIT HERE ===============



        # =========================================

        return dz

class GELU:
    """
    GELU (Gaussian Error Linear Unit) Function using sigmoid approximation.
    GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) ≈ x * sigmoid(1.702 * x)
    Implement forward & backward path of GELU.
    """
    def __init__(self):
        self.x = None
        self.sigmoid_term = None
        self.sigmoid = Sigmoid()
        
    def forward(self, z):
        """
        GELU Forward using sigmoid approximation.
        GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))) ≈ x * sigmoid(1.702 * x)

        z --> (GELU) --> out

        [Inputs]
            z : GELU input in any shape.

        [Outputs]
            out : GELU(z).
        """
        out = None
        
        # Store input for backward pass
        self.x = z.copy()
        self.sigmoid_term = self.sigmoid.forward(1.702 * z)
        # Calculate GELU using sigmoid approximation
        out = z * self.sigmoid_term
        
        return out

    def backward(self, d_prev):
        """
        GELU Backward using sigmoid approximation.

        z --> (GELU) --> out
        dz <-- (dGELU) <-- d_prev(dL/dout)

        [Inputs]
            d_prev : Gradients until now.
            d_prev = dL/dk, where k = GELU(z).

        [Outputs]
            dz : Gradients w.r.t. GELU input z.
        """
        dgelu = None


        # =============== EDIT HERE ===============

        

        # =========================================
        
        return  dgelu

class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self, z):
        """
        Sigmoid Forward.

        z --> (Sigmoid) --> self.out

        [Inputs]
            z : Sigmoid input in any shape.

        [Outputs]
            self.out : Sigmoid(z).
        """

        self.out = None
        self.out = 1 / (1 + np.exp(-z))
        
        return self.out

    def backward(self, d_prev):
        """
        Sigmoid Backward.

        z --> (Sigmoid) --> self.out
        dz <-- (dSigmoid) <-- d_prev(dL/d self.out)

        [Inputs]
            d_prev : Gradients until now.

        [Outputs]
            dz : Gradients w.r.t. Sigmoid input z which is self.out.
        """

        dz = None
        # =============== EDIT HERE ===============




        # =========================================
        return dz
    
class Linear:
    def __init__(self, num_hidden_1, num_hidden_2):
        limit = np.sqrt(2 / float(num_hidden_1))
        self.W = np.random.normal(0.0, limit, size=(num_hidden_1, num_hidden_2))
        self.b = np.zeros(num_hidden_2)

        self.dW = None
        self.db = None

    def forward(self, x):
        
        """
        Linear layer forward
        - Feed forward
        - Apply activation function you implemented above.

        [Inputs]
           x : Input data (N, D)

        [Outputs]
            self.out : Output of Linear layer. Hidden. (N, H)
        """

        self.x = x
        
        self.out = np.matmul(self.x, self.W) + self.b

        return self.out

    def backward(self, d_prev):
        """
        Linear layer backward
        x and (W & b) --> z -- (activation) --> hidden
        dx and (dW & db) <-- dz <-- (activation) <-- hidden

        - Backward of activation
        - Gradients of W, b

        [Inputs]
            d_prev : Gradients until now.

        [Outputs]
            dx : Gradients of input x
        """
        dx = None
        self.dW = None
        self.db = None
        # =============== EDIT HERE ===============



        # =========================================
        
        return dx
    


class SigmoidLayer:
    def __init__(self, num_hidden_2, num_outputs):
        self.x = None
        self.y = None
        self.y_hat = None

        self.loss = None

        self.sigmoid = Sigmoid()
        self.linear = Linear(num_hidden_2, num_outputs)

    def forward(self, x, y):
        """
        Sigmoid output layer forward
        - Make prediction
        - Calculate loss

        """
        self.y_hat = self.predict(x)
        self.y = y
        self.x = x

        self.loss = self.binary_ce_loss(self.y_hat, self.y)

        return self.loss

    def binary_ce_loss(self, y_hat, y):
        """
        Calcualte "Binary cross-entropy loss"
        Add 'eps' for stability inside log function.

        [Inputs]
            y_hat : Prediction
            y : Label

        [Outputs]
            loss value
        """
        eps = 1e-10
        bce_loss = None

        y = np.asarray(y).reshape(y_hat.shape)
        bce_loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

        return bce_loss

    def predict(self, x):
        """
        Make prediction in probability. (Not 0 or 1 label!!)

        [Inputs]
            x : input data

        [Outputs]
            y_hat : Prediction
        """
  
        z = self.linear.forward(x)
        y_hat = self.sigmoid.forward(z)
        
        return y_hat

    def backward(self, d_prev=1):
        """
        Calculate gradients of input (x), W, b of this layer.
        Save self.dW, self.db to update later.

        x and (W & b) --> z -- (activation) --> y_hat --> Loss
        dx and (dW & db) <-- dz <-- (activation) <-- dy_hat <-- Loss

        [Inputs]
            d_prev : Gradients until here. (Always 1 since its output layer)

        [Outputs]
            dx : Gradients of output layer input x (Not MLP input x!)
        """
        batch_size = self.y.shape[0]
        eps = 1e-10
        y = self.y.reshape(self.y_hat.shape)
        d_bce = (-y / (self.y_hat + eps)
                 + (1 - y) / (1 - self.y_hat + eps)) / batch_size
        
        sig_d_prev= d_prev * d_bce
        lin_d_prev = None

        dx = None
        # =============== EDIT HERE ===============
        """
        you should calculate gradient of sigmoid and linear layer
        ! tip: use backward functions that you have already written!!!! 
        """
        # call backward 



        # =========================================
        
        self.dW = self.linear.dW
        self.db = self.linear.db
        return dx
    



