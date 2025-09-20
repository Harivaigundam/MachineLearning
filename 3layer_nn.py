import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

# -------------------- Layers --------------------
class Dense_Layer:
    def __init__(self, n_inputs, n_neurons, w_r_l1=0, w_r_l2=0, b_r_l1=0, b_r_l2=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.w_r_l1 = w_r_l1
        self.w_r_l2 = w_r_l2
        self.b_r_l1 = b_r_l1
        self.b_r_l2 = b_r_l2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # L1 on weights
        if self.w_r_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.w_r_l1 * dL1

        # L1 on biases
        if self.b_r_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.b_r_l1 * dL1

        # L2 on weights
        if self.w_r_l2 > 0:
            self.dweights += 2 * self.w_r_l2 * self.weights

        # L2 on biases
        if self.b_r_l2 > 0:
            self.dbiases += 2 * self.b_r_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


class Loss:
    def regularization_loss(self, layer):
        reg_loss = 0
        if layer.w_r_l1 > 0:
            reg_loss += layer.w_r_l1 * np.sum(np.abs(layer.weights))
        if layer.b_r_l1 > 0:
            reg_loss += layer.b_r_l1 * np.sum(np.abs(layer.biases))
        if layer.w_r_l2 > 0:
            reg_loss += layer.w_r_l2 * np.sum(layer.weights * layer.weights)
        if layer.b_r_l2 > 0:
            reg_loss += layer.b_r_l2 * np.sum(layer.biases * layer.biases)
        return reg_loss

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)


class CategoricalEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        return -np.log(correct_confidences)


class Activation_softmax_crossEntropyLoss:
    def __init__(self):
        self.softmax = Softmax()
        self.loss_fn = CategoricalEntropyLoss()

    def forward(self, inputs, y_true):
        self.softmax.forward(inputs)
        self.output = self.softmax.output
        return self.loss_fn.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Optimiser_ADAM:
    def __init__(self, learning_rate=0.02, decay=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_momentum'):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Momentum update
        layer.weight_momentum = self.beta_1 * layer.weight_momentum + (1 - self.beta_1) * layer.dweights
        layer.bias_momentum = self.beta_1 * layer.bias_momentum + (1 - self.beta_1) * layer.dbiases
        weight_mom_crt = layer.weight_momentum / (1 - (self.beta_1 ** (self.iterations + 1)))
        bias_mom_crt = layer.bias_momentum / (1 - (self.beta_1 ** (self.iterations + 1)))

        # Cache update
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * (layer.dweights ** 2)
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * (layer.dbiases ** 2)
        weight_cache_crt = layer.weight_cache / (1 - (self.beta_2 ** (self.iterations + 1)))
        bias_cache_crt = layer.bias_cache / (1 - (self.beta_2 ** (self.iterations + 1)))

        # Weight update
        layer.weights += -self.current_learning_rate * weight_mom_crt / (np.sqrt(weight_cache_crt) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_mom_crt / (np.sqrt(bias_cache_crt) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Dropout:
    def __init__(self,rate):
        self.rate=1-rate

    def forward(self, inputs):
        self.inputs=inputs
        self.binarymask= np.random.binomial(1,self.rate,size=inputs.shape)/self.rate
        self.output=self.inputs*self.binarymask

    def backward(self, dvalues):
        self.dinputs = dvalues*self.binarymask


# -------------------- Training --------------------
X, y = spiral_data(samples=100, classes=3)

dense1 = Dense_Layer(2, 128, w_r_l2=1e-4)
act1 = Activation_Relu()
dropout1=Dropout(0.1)

dense2 = Dense_Layer(128, 64, w_r_l2=1e-4)
act2 = Activation_Relu()
dropout2=Dropout(0.1)

dense3 = Dense_Layer(64, 3)
loss_activation = Activation_softmax_crossEntropyLoss()

optimizer = Optimiser_ADAM(learning_rate=0.02, decay=1e-5)

train_losses = []
train_accuracies = []

for iteration in range(10001):
    # Forward
    dense1.forward(X)
    act1.forward(dense1.output)
    dropout1.forward(act1.output)

    dense2.forward(dropout1.output)
    act2.forward(dense2.output)
    dropout2.forward(act2.output)

    dense3.forward(dropout2.output)
    data_loss = loss_activation.forward(dense3.output, y)

    reg_loss = loss_activation.loss_fn.regularization_loss(dense1) + \
               loss_activation.loss_fn.regularization_loss(dense2)
    loss = data_loss + reg_loss

    predictions = np.argmax(loss_activation.output, axis=1)
    acc = np.mean(predictions == y)

    train_losses.append(loss)
    train_accuracies.append(acc)

    if iteration % 1000 == 0:
        print(f"iter={iteration}, loss={loss:.3f}, acc={acc:.3f}")

    # Backward
    loss_activation.backward(loss_activation.output, y)
    dense3.backward(loss_activation.dinputs)
    dropout2.backward(dense3.dinputs)
    act2.backward(dropout2.dinputs)
    dense2.backward(act2.dinputs)
    dropout1.backward(dense2.dinputs)
    act1.backward(dropout1.dinputs)
    dense1.backward(act1.dinputs)

    # Update
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

# -------------------- Plot Loss/Accuracy --------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.show()

# -------------------- Decision Boundary Plot --------------------
def plot_decision_boundary(X, y, model_layers, step=0.02):
    # Create grid of points
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Forward pass on grid points
    model_layers[0].forward(grid)
    model_layers[1].forward(model_layers[0].output)
    model_layers[2].forward(model_layers[1].output)
    model_layers[3].forward(model_layers[2].output)
    model_layers[4].forward(model_layers[3].output)

    # Softmax for probabilities
    sm = Softmax()
    sm.forward(model_layers[4].output)

    Z = np.argmax(sm.output, axis=1)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title("Decision Boundary")

plot_decision_boundary(X, y, [dense1, act1, dense2, act2, dense3])
plt.show()

# -------------------- Test --------------------
X_test, y_test = spiral_data(samples=100, classes=3)

dense1.forward(X_test)
act1.forward(dense1.output)
dense2.forward(act1.output)
act2.forward(dense2.output)
dense3.forward(act2.output)
loss = loss_activation.forward(dense3.output, y_test)

preds = np.argmax(loss_activation.output, axis=1)
acc = np.mean(preds == y_test)
print(f"Test loss={loss:.3f}, acc={acc:.3f}")

plot_decision_boundary(X_test, y_test, [dense1, act1, dense2, act2, dense3])
plt.show()