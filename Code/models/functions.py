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
        # z = activation에 들어오는 input. 바로 앞 layer의 output.
        # out = activation의 output, f(z). 여기서 f()는 ReLU 함수.
        # d_prev = dL/dout. 오른쪽 layer가 넘겨주는 값.
        # dz = dL/dz. 내가 구하고 왼쪽 layer에 넘겨줄 값.
        # self.zero_mask = z가 0보다 작을경우 true, 0보다 크거나 같을 경우 false.
        # z = np.array([[ 3, -1],
        #               [-2,  5]])
        # zero_mask = z < 0
        # [[False,  True],
        #  [ True, False]]

        # dz = dL/dout * dout/dz
        #    = d_prev * f'(z)
        dfdz = ~self.zero_mask # false와 true를 반대로 뒤집음. 

        dz = d_prev * dfdz # true일 경우 1, false일 경우 0이므로, 음수는 0으로 바뀌고 양수는 그대로. 

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
        # z = activation에 들어오는 input. 바로 앞 layer의 output.
        # out = activation의 output, f(z). 여기서 f()는 ReLU 함수.
        # d_prev = dL/dout. 오른쪽 layer가 넘겨주는 값.
        # dz = dL/dz. 내가 구하고 왼쪽 layer에 넘겨줄 값. AKA. dgelu
        # self.sigmod_term = sigmoid값.
        # GELU의 sigmoid = sigmoid(1.702 * z). 
        #   따라서 미분 = sigmoid(1.702 * z) + 1.702 * z * sigmoid(1.702 * z) * (1 - sigmoid(1.702 * z))

        dsigmoid = self.sigmoid_term + 1.702 * self.x * self.sigmoid_term * (1 - self.sigmoid_term)
        dgelu = d_prev * dsigmoid        

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
        # z = activation에 들어오는 input. 바로 앞 layer의 output.
        # out = activation의 output, f(z). 여기서 f()는 ReLU 함수.
        # d_prev = dL/dout. 오른쪽 layer가 넘겨주는 값.
        # dz = dL/dz. 내가 구하고 왼쪽 layer에 넘겨줄 값.
        # self.out = sigmoid값.

        dsigmoid = self.out * (1 - self.out)
        dz = d_prev * dsigmoid
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
        # z = activation에 들어오는 input. 바로 앞 layer의 output.
        # d_prev = dL/dout. 오른쪽 layer가 넘겨주는 값.
        # self.dW = dL/dW. 내가 구하고 update할 값.
        # self.db = dL/db. 내가 구하고 update할 값.
        # dx = dL/dx. 내가 구하고 왼쪽 layer에 넘겨줄 값.
        # self.out = linear layer output 값. out = xW + b

        # dout/dW = x.T
        # x = [N, D], d_prev = [N, H], W = [D, H]
        self.dW = np.matmul(self.x.T, d_prev) # dW = [D, H]
        # dout/db = 1
        # d_prev = [N, H], b = [H,]
        self.db = np.sum(d_prev, axis=0) # db = [H,]
        # dout/dx = W.T
        # d_prev = [N, H], W = [D, H]
        dx = np.matmul(d_prev, self.W.T) # dx = [N, D]
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
        # self.x = activation에 들어오는 input. 바로 앞 layer의 output.
        # self.y_hat = 예측 확률 sigmoid(z)
        # self.y = 정답 label
        # d_bce = dL/dy_hat. binary cross entropy loss에 대한 gradient. 이미 계산됨.
        # sig_d_prev = d_prev * d_bce. sigmoid layer에 대한 gradient. sigmoid backward로 넘겨줄 값.
        # lin_d_prev = linear layer에 대한 gradient. linear backward로 넘겨줄 값. 
        # dx = 다음 layer로 넘겨줄 gradient. dL/dx.

        # 구조:
        # x -> [self.linear] -> z -> [self.sigmoid] -> y_hat -> loss
        #      Linear 객체            Sigmoid 객체

        # bce(binary cross entropy):
        # L = -1/N * sum(y * log(y_hat) + (1 - y) * log(1 - y_hat))
        # 그리고 bce의 미분은 다음과 같이 계산됨:
        # dL/dy_hat = 1/N * (-y/y_hat + (1 - y)/(1-y_hat))

        # 값을 두 번 넘겨야 한다:
        # sig_d_prev -> self.sigmoid.backward -> lin_d_prev -> self.linear.backward -> dx
        # 1. self.sigmoid의 backward에 sig_d_prev를 넣고, 결과를 lin_d_prev가 받는다.
        # 2. self.linear의 backward에 lin_d_prev에 넣고, 결과를 dx가 받는다.

        # chain rule 전개:
        # dL/dx = dL/dy_hat * dy_hat/dz * dz/dx
        #
        # (1) dL/dy_hat = d_prev * d_bce                      -> sig_d_prev
        #     (d_prev = 1 이므로 sig_d_prev = d_bce)
        # (2) dL/dz     = dL/dy_hat * dy_hat/dz
        #               = sig_d_prev * y_hat * (1 - y_hat)    -> lin_d_prev  [self.sigmoid.backward]
        # (3) dL/dW     = x.T @ dL/dz                         -> self.linear.dW  [self.linear.backward]
        #     dL/db     = sum(dL/dz, axis=0)                  -> self.linear.db
        #     dL/dx     = dL/dz @ W.T                         -> dx
        #
        # 각 gradient를 담은 variable:
        # dL/dy_hat = sig_d_prev (= d_bce)
        # dy_hat/dz = y_hat * (1 - y_hat). 따로 저장 안 됨. Sigmoid.backward 안의 dsigmoid.
        # dL/dz     = lin_d_prev
        # dz/dW, dz/db, dz/dx = x.T, 1, W.T. 따로 저장 안 됨. Linear.backward 안에서 곱해짐.
        # dL/dW     = self.linear.dW (-> 아래에서 self.dW로 복사)
        # dL/db     = self.linear.db (-> 아래에서 self.db로 복사)
        # dL/dx     = dx

        lin_d_prev = self.sigmoid.backward(sig_d_prev) # 구조에서 z로 미분한 것. dL/dz.
        dx = self.linear.backward(lin_d_prev) 
        # =========================================
        
        self.dW = self.linear.dW
        self.db = self.linear.db
        return dx
    



