import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

torch.manual_seed(42)
np.random.seed(42)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch1 = unpickle('cifar-10-batches-py/data_batch_1')
batch2 = unpickle('cifar-10-batches-py/data_batch_2')
batch3 = unpickle('cifar-10-batches-py/data_batch_3')
batch4 = unpickle('cifar-10-batches-py/data_batch_4')
batch5 = unpickle('cifar-10-batches-py/data_batch_5')
test_batch = unpickle('cifar-10-batches-py/test_batch')

train_data = np.vstack([batch1[b'data'], batch2[b'data'], batch3[b'data'], batch4[b'data'], batch5[b'data']])
train_labels = np.hstack([batch1[b'labels'], batch2[b'labels'], batch3[b'labels'], batch4[b'labels'], batch5[b'labels']])

test_data = test_batch[b'data']
test_labels = np.array(test_batch[b'labels'])

def prepare_data(data):
    data = data.astype(np.float32) / 255.0
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    return data

X_train = prepare_data(train_data)
y_train = train_labels.astype(np.int64)
X_test = prepare_data(test_data) 
y_test = test_labels.astype(np.int64)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

class NN(nn.Module):
    def __init__(self, input_size=3072, hidden_size=128, num_classes=10):
        super(NN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)
    
    
def compute_sharpness(model, X, y, loss_fn):
    
    model.zero_grad()
    output = model(X)
    loss = loss_fn(output, y)
    params = [p for p in model.parameters() if p.requires_grad]

    first_grad = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.reshape(-1) for g in first_grad])

    v = torch.randn_like(grad_flat)
    v = v / torch.norm(v)
    num_iterations = 10

    for i in range(num_iterations):
        Hv = torch.autograd.grad(
            torch.dot(grad_flat, v),
            params,
            retain_graph=True,
            allow_unused=True
        )
        Hv_valid = [h for h in Hv if h is not None]
        if not Hv_valid: return 0.0
        Hv_flat = torch.cat([h.reshape(-1) for h in Hv_valid])
        v = Hv_flat / torch.norm(Hv_flat)

    Hv_final = torch.autograd.grad(torch.dot(grad_flat, v), params, retain_graph=True, allow_unused=True)
    Hv_final_valid = [h for h in Hv_final if h is not None]
    if not Hv_final_valid: return 0.0
    Hv_final_flat = torch.cat([h.reshape(-1) for h in Hv_final_valid])
    lambda_max = torch.dot(v, Hv_final_flat).item()
    
    return abs(lambda_max)


learning_rates = [0.05, 0.1]

plt.figure(figsize=(14, 6))

for eta in learning_rates:

    model_name, model = ("NN with SGD and Cross-Entropy Loss", NN(hidden_size=256))
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = eta
    optimiser = optim.SGD(model.parameters(), lr=learning_rate)
    n_epochs = 500

    subset_size = 2000
    indices = torch.randperm(len(X_train))[:subset_size]
    X_small = X_train[indices]
    y_small = y_train[indices]

    change_in_loss = []
    change_in_sharpness = []
    EoS_limit = 2 / learning_rate

    for epoch in range(n_epochs):
        y_pred = model(X_small)
        loss = loss_fn(y_pred, y_small)
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        change_in_loss.append(loss.item())
        
        if epoch % 10 == 0 or epoch < 5 or epoch > n_epochs - 5:
            try:
                lambda_max = compute_sharpness(model, X_small, y_small, loss_fn)
                change_in_sharpness.append((epoch, lambda_max))
            except Exception as e:
                print(f"Sharpness failed at epoch {epoch}: {e}")
                change_in_sharpness.append((epoch, 0.0))

    plt.subplot(1, 2, 1)
    plt.plot(change_in_loss, label=rf'$\lambda_{{\max}} = {eta}$')
    plt.title(f'{model_name} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    if change_in_sharpness:
        epochs = [s[0] for s in change_in_sharpness]
        sharpness = [s[1] for s in change_in_sharpness]
        plt.plot(epochs, sharpness, '-', label=rf'$\lambda_{{\max}} = {eta}$')
        plt.axhline(y=EoS_limit, color='r', linestyle='--', label=f'EoS limit: {EoS_limit:.1f}')
        plt.title(f'{model_name} - Sharpness Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Sharpness')
        plt.legend()
        plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
