import argparse
import wandb
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Feedforward Neural Network Training Script")
    
    parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname', help='Project name used to track experiments in W&B')
    parser.add_argument('-we', '--wandb_entity', type=str, default='myname', help='Wandb Entity used to track experiments')
    
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=['mnist', 'fashion_mnist'], help='Dataset choice')
    
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'], help='Loss function')
    parser.add_argument('-o', '--optimizer', type=str, default='nadam', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], help='Optimizer')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum for momentum and nag')
    parser.add_argument('-beta', '--beta', type=float, default=0.9, help='Beta for rmsprop')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9, help='Beta1 for adam/nadam')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.999, help='Beta2 for adam/nadam')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-8, help='Epsilon for optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help='Weight decay (L2 penalty)')
    
    parser.add_argument('-w_i', '--weight_init', type=str, default='random', choices=['random', 'Xavier'], help='Weight initialization method')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, default=64, help='Number of hidden neurons per layer')
    parser.add_argument('-a', '--activation', type=str, default='tanh', choices=['identity', 'sigmoid', 'tanh', 'ReLU'], help='Activation function')
    
    return parser.parse_args()



def build_params(sizes, init):
    params = []
    np.random.seed(42) 
    for i in range(len(sizes)-1):
        if init.lower() == 'random':
            w = np.random.randn(sizes[i], sizes[i+1]) * 0.01
        elif init.lower() == 'xavier':
            L = np.sqrt(6 / (sizes[i] + sizes[i+1]))
            w = np.random.uniform(-L, L, (sizes[i], sizes[i+1]))
        b = np.zeros(sizes[i+1])
        params.append({'w': w, 'b': b})
    return params

def softmax(x):
    x_norm = x - np.max(x, axis=1, keepdims=True)
    return np.exp(x_norm) / np.exp(x_norm).sum(axis=1, keepdims=True)

def forward_pass(data, params, a_fn):
    a = [data]
    z = []
    for i in range(len(params)):
        z.append(a[i] @ params[i]['w'] + params[i]['b'])
        if i == len(params) - 1:
            a.append(softmax(z[i]))
        else:
            if a_fn.lower() == 'sigmoid':
                a.append(1 / (1 + np.exp(-z[i])))
            elif a_fn.lower() == 'tanh':
                a.append(np.tanh(z[i]))
            elif a_fn.lower() == 'relu':
                a.append(np.maximum(0, z[i]))
            elif a_fn.lower() == 'identity':
                a.append(z[i])
    return a, z

def loss_func(output, labels, fn):
    one_hot = np.eye(10)[labels]
    N = output.shape[0]
    if fn.lower() == 'cross_entropy':
        true_class_probs = (output * one_hot).sum(axis=1)
        loss_val = np.mean(-np.log(true_class_probs + 1e-9))
        grad = (output - one_hot) / N
        return loss_val, grad
    elif fn.lower() in ['mse', 'mean_squared_error']:
        loss_val = np.mean(np.sum((output - one_hot)**2, axis=1))
        diff = output - one_hot
        s = np.sum(diff * output, axis=1, keepdims=True)
        grad = (2.0 / N) * output * (diff - s)
        return loss_val, grad

def activation_derivative(a, a_fn):
    if a_fn.lower() == 'sigmoid':
        return a * (1 - a)
    elif a_fn.lower() == 'tanh':
        return 1 - (a**2)
    elif a_fn.lower() == 'relu':
        return np.where(a > 0, 1.0, 0.0)
    elif a_fn.lower() == 'identity':
        return np.ones_like(a)

def backprop(output_gradient, params, a, z, a_fn):
    grads = [None] * len(params)
    dz = output_gradient
    for i in reversed(range(len(params))):
        dw = a[i].T @ dz
        db = dz.sum(axis=0)
        grads[i] = {'dw': dw, 'db': db}
        if i > 0:
            dz = (dz @ params[i]['w'].T) * activation_derivative(a[i], a_fn)
    return grads

def accuracy(params, X, Y, a_fn):
    a, _ = forward_pass(X, params, a_fn)
    preds = np.argmax(a[-1], axis=1)     
    return np.mean(preds == Y)


class SGD:
    def __init__(self, lr, weight_decay=0.0):
        self.lr = lr
        self.wd = weight_decay
    def step(self, params, grads):
        for i in range(len(params)):
            params[i]['w'] -= self.lr * (grads[i]['dw'] + self.wd * params[i]['w'])
            params[i]['b'] -= self.lr * grads[i]['db']

class Momentum:
    def __init__(self, lr, m, params, weight_decay=0.0):
        self.lr = lr
        self.m = m
        self.wd = weight_decay
        self.velocity = [{'w': np.zeros_like(p['w']), 'b': np.zeros_like(p['b'])} for p in params]
    def step(self, params, grads):
        for i in range(len(params)):
            dw = grads[i]['dw'] + self.wd * params[i]['w']
            self.velocity[i]['w'] = self.m * self.velocity[i]['w'] + dw
            self.velocity[i]['b'] = self.m * self.velocity[i]['b'] + grads[i]['db']
            params[i]['w'] -= self.lr * self.velocity[i]['w']
            params[i]['b'] -= self.lr * self.velocity[i]['b']

class Nesterov:
    def __init__(self, lr, m, params, weight_decay=0.0):
        self.lr = lr
        self.m = m
        self.wd = weight_decay
        self.velocity = [{'w': np.zeros_like(p['w']), 'b': np.zeros_like(p['b'])} for p in params]
    def step(self, params, grads):
        for i in range(len(params)):
            dw = grads[i]['dw'] + self.wd * params[i]['w']
            self.velocity[i]['w'] = self.m * self.velocity[i]['w'] + dw
            self.velocity[i]['b'] = self.m * self.velocity[i]['b'] + grads[i]['db']
            params[i]['w'] -= self.lr * (self.m * self.velocity[i]['w'] + dw)
            params[i]['b'] -= self.lr * (self.m * self.velocity[i]['b'] + grads[i]['db'])

class RMSProp:
    def __init__(self, lr, beta, params, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.wd = weight_decay
        self.v = [{'w': np.zeros_like(p['w']), 'b': np.zeros_like(p['b'])} for p in params]
    def step(self, params, grads):
        for i in range(len(params)):
            dw = grads[i]['dw'] + self.wd * params[i]['w']
            self.v[i]['w'] = self.beta * self.v[i]['w'] + (1 - self.beta) * dw**2
            self.v[i]['b'] = self.beta * self.v[i]['b'] + (1 - self.beta) * grads[i]['db']**2
            params[i]['w'] -= self.lr * grads[i]['dw'] / (np.sqrt(self.v[i]['w']) + self.eps)
            params[i]['b'] -= self.lr * grads[i]['db'] / (np.sqrt(self.v[i]['b']) + self.eps)

class Adam:
    def __init__(self, lr, beta1, beta2, params, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1            
        self.beta2 = beta2            
        self.eps = eps
        self.t = 0
        self.wd = weight_decay
        self.m = [{'w': np.zeros_like(p['w']), 'b': np.zeros_like(p['b'])} for p in params]
        self.v = [{'w': np.zeros_like(p['w']), 'b': np.zeros_like(p['b'])} for p in params]
    def step(self, params, grads):
        self.t += 1
        for i in range(len(params)):
            for key, gkey in [('w', 'dw'), ('b', 'db')]:
                g = grads[i][gkey]
                if key == 'w': g = g + self.wd * params[i]['w']
                self.m[i][key] = self.beta1 * self.m[i][key] + (1 - self.beta1) * g
                self.v[i][key] = self.beta2 * self.v[i][key] + (1 - self.beta2) * g**2
                m_hat = self.m[i][key] / (1 - self.beta1**self.t)
                v_hat = self.v[i][key] / (1 - self.beta2**self.t)
                params[i][key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class Nadam:
    def __init__(self, lr, beta1, beta2, params, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.wd = weight_decay
        self.m = [{'w': np.zeros_like(p['w']), 'b': np.zeros_like(p['b'])} for p in params]
        self.v = [{'w': np.zeros_like(p['w']), 'b': np.zeros_like(p['b'])} for p in params]
    def step(self, params, grads):
        self.t += 1
        for i in range(len(params)):
            for key, gkey in [('w', 'dw'), ('b', 'db')]:
                g = grads[i][gkey]
                if key == 'w': g = g + self.wd * params[i]['w']
                self.m[i][key] = self.beta1 * self.m[i][key] + (1 - self.beta1) * g
                self.v[i][key] = self.beta2 * self.v[i][key] + (1 - self.beta2) * g**2
                m_hat = self.m[i][key] / (1 - self.beta1**self.t)
                v_hat = self.v[i][key] / (1 - self.beta2**self.t)
                m_nesterov = self.beta1 * m_hat + (1 - self.beta1) * g / (1 - self.beta1**self.t)
                params[i][key] -= self.lr * m_nesterov / (np.sqrt(v_hat) + self.eps)

def make_optimizer(args, params):
    name = args.optimizer.lower()
    if name == 'sgd':      return SGD(args.learning_rate, weight_decay=args.weight_decay)
    if name == 'momentum': return Momentum(args.learning_rate, args.momentum, params, weight_decay=args.weight_decay)
    if name == 'nag':      return Nesterov(args.learning_rate, args.momentum, params, weight_decay=args.weight_decay)
    if name == 'rmsprop':  return RMSProp(args.learning_rate, args.beta, params, eps=args.epsilon, weight_decay=args.weight_decay)
    if name == 'adam':     return Adam(args.learning_rate, args.beta1, args.beta2, params, eps=args.epsilon, weight_decay=args.weight_decay)
    if name == 'nadam':    return Nadam(args.learning_rate, args.beta1, args.beta2, params, eps=args.epsilon, weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")



def main():
    args = parse_args()
    
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    if args.dataset == 'fashion_mnist':
        from keras.datasets import fashion_mnist as keras_dataset
    else:
        from keras.datasets import mnist as keras_dataset
        
    (X_train_raw, Y_train_raw), (X_test_raw, Y_test_raw) = keras_dataset.load_data()
    
    X_train_flat = X_train_raw.reshape(X_train_raw.shape[0], -1)/255.0
    X_test_flat = X_test_raw.reshape(X_test_raw.shape[0], -1)/255.0
    
    # 10% Validation Split
    np.random.seed(42)
    perm = np.random.permutation(len(X_train_flat))
    X_train_shuffled = X_train_flat[perm]
    Y_train_shuffled = Y_train_raw[perm]
    
    split_idx = int(0.1 * len(X_train_flat))
    X_val = X_train_shuffled[:split_idx]
    Y_val = Y_train_shuffled[:split_idx]
    X_train = X_train_shuffled[split_idx:]
    Y_train = Y_train_shuffled[split_idx:]
    
    sizes = [784] + [args.hidden_size] * args.num_layers + [10]
    params = build_params(sizes, args.weight_init)
    opt = make_optimizer(args, params)
    
    N = len(X_train)
    for epoch in range(args.epochs):
        
        perm = np.random.permutation(N)
        X_sh = X_train[perm]
        Y_sh = Y_train[perm]

        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, N, args.batch_size):
            bX = X_sh[start:start + args.batch_size]
            bY = Y_sh[start:start + args.batch_size]

            a, z = forward_pass(bX, params, args.activation)
            batch_loss, output_grad = loss_func(a[-1], bY, args.loss)
            grads = backprop(output_grad, params, a, z, args.activation)
            opt.step(params, grads)

            epoch_loss += batch_loss
            n_batches += 1
            
        va_tr, _ = forward_pass(X_train, params, args.activation)
        train_loss, _ = loss_func(va_tr[-1], Y_train, args.loss)
        train_acc = accuracy(params, X_train, Y_train, args.activation)
        
        va_val, _ = forward_pass(X_val, params, args.activation)
        val_loss, _ = loss_func(va_val[-1], Y_val, args.loss)
        val_acc = accuracy(params, X_val, Y_val, args.activation)

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        })
        
        print(f"Epoch {epoch+1:2d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")



main()