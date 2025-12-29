import numpy as np
from utils import im2col, col2im

class Conv:
    def __init__(self, in_c, out_c, f_size, stride=1, pad=1):
        # He initialization
        self.w = np.random.randn(out_c, in_c, f_size, f_size) * np.sqrt(2 / (in_c * f_size * f_size))
        self.b = np.zeros((out_c, 1))
        self.stride, self.pad = stride, pad
        
        # Adam Optimizer Parametreleri
        self.m_w, self.v_w = np.zeros_like(self.w), np.zeros_like(self.w)
        self.m_b, self.v_b = np.zeros_like(self.b), np.zeros_like(self.b)
        self.t = 0

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        self.col = im2col(x, self.w.shape[2], self.w.shape[3], self.pad, self.stride)
        w_col = self.w.reshape(self.w.shape[0], -1)
        out = w_col @ self.col + self.b
        out_h = int((H + 2 * self.pad - self.w.shape[2]) / self.stride + 1)
        out_w = int((W + 2 * self.pad - self.w.shape[3]) / self.stride + 1)
        return out.reshape(self.w.shape[0], out_h, out_w, N).transpose(3, 0, 1, 2)

    def backward(self, dout, lr):
        N, out_c, H, W = dout.shape
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(out_c, -1)
        dw = (dout_reshaped @ self.col.T).reshape(self.w.shape)
        db = np.sum(dout, axis=(0, 2, 3)).reshape(-1, 1)
        
        w_flat = self.w.reshape(out_c, -1)
        dcol = w_flat.T @ dout_reshaped
        dx = col2im(dcol, self.x.shape, self.w.shape[2], self.w.shape[3], self.pad, self.stride)
        
        # --- Adam Optimizer Güncellemesi ---
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        # Ağırlıklar (Weights)
        self.m_w = beta1 * self.m_w + (1 - beta1) * dw
        self.v_w = beta2 * self.v_w + (1 - beta2) * (dw**2)
        m_hat = self.m_w / (1 - beta1**self.t)
        v_hat = self.v_w / (1 - beta2**self.t)
        self.w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Yanlılık (Bias)
        self.m_b = beta1 * self.m_b + (1 - beta1) * db
        self.v_b = beta2 * self.v_b + (1 - beta2) * (db**2)
        mb_hat = self.m_b / (1 - beta1**self.t)
        vb_hat = self.v_b / (1 - beta2**self.t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
        
        return dx

class MaxPool:
    def forward(self, x):
        self.x_shape = x.shape
        N, C, H, W = x.shape
        out = x[:, :, :H//2*2, :W//2*2].reshape(N, C, H//2, 2, W//2, 2).max(axis=(3, 5))
        self.arg_max = (x == out.repeat(2, axis=2).repeat(2, axis=3))
        return out

    def backward(self, dout, lr):
        dx = dout.repeat(2, axis=2).repeat(2, axis=3)
        return (dx * self.arg_max)[:, :, :self.x_shape[2], :self.x_shape[3]]

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, x * self.alpha)
    def backward(self, dout, lr):
        dx = dout.copy()
        dx[self.x <= 0] *= self.alpha
        return dx

class Dense:
    def __init__(self, in_dim, out_dim):
        self.w = np.random.randn(in_dim, out_dim) * np.sqrt(2 / in_dim)
        self.b = np.zeros((1, out_dim))
        
        # Adam Optimizer Parametreleri
        self.m_w, self.v_w = np.zeros_like(self.w), np.zeros_like(self.w)
        self.m_b, self.v_b = np.zeros_like(self.b), np.zeros_like(self.b)
        self.t = 0

    def forward(self, x):
        self.original_shape = x.shape
        self.x_flat = x.reshape(x.shape[0], -1)
        return self.x_flat @ self.w + self.b

    def backward(self, dout, lr):
        dw = self.x_flat.T @ dout
        db = np.sum(dout, axis=0, keepdims=True)
        dx_flat = dout @ self.w.T
        
        # --- Adam Optimizer Güncellemesi ---
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        # Ağırlıklar
        self.m_w = beta1 * self.m_w + (1 - beta1) * dw
        self.v_w = beta2 * self.v_w + (1 - beta2) * (dw**2)
        m_hat = self.m_w / (1 - beta1**self.t)
        v_hat = self.v_w / (1 - beta2**self.t)
        self.w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Yanlılık
        self.m_b = beta1 * self.m_b + (1 - beta1) * db
        self.v_b = beta2 * self.v_b + (1 - beta2) * (db**2)
        mb_hat = self.m_b / (1 - beta1**self.t)
        vb_hat = self.v_b / (1 - beta2**self.t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
        
        return dx_flat.reshape(self.original_shape)

class BloodCellAI:
    def __init__(self):
        self.layers = [
            Conv(3, 32, 3, pad=1),   # 0
            LeakyReLU(),             # 1
            MaxPool(),               # 2
            Conv(32, 64, 3, pad=1),  # 3
            LeakyReLU(),             # 4
            MaxPool(),               # 5
            Dense(64 * 7 * 7, 128),  # 6
            LeakyReLU(),             # 7
            Dense(128, 8)            # 8
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shift_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)