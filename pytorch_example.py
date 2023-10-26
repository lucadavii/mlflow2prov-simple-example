import torch
from torch import nn
import mlflow

MANUAL_SEED=42

#known params
WEIGHT = 0.7
BIAS = 0.3

#line
START = 0; END = 1; STEP = 0.02

#data 
X = torch.arange(START,END,STEP).unsqueeze(dim=1)
#labels
y = WEIGHT *X + BIAS

train_split = int(0.8*len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# def plot_predictions (train_data = X_train, train_labels = y_train, test_data = X_test, test_labels = y_test, predictions = None):
#     plt.figure(figsize=(10,7))
#     plt.scatter(train_data,train_labels, c="b", s=4, label="Training data")
#     plt.scatter(test_data,test_labels, c="g", s=4, label="Testing data")

#     if predictions is not None:
#         plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
#     plt.legend(prop={"size":14})
#     plt.show()
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.manual_seed(MANUAL_SEED)
with mlflow.start_run() as run:
    model_0 = LinearRegressionModel()
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.01)
    
    EPOCHS = 100
    mlflow.log_param("epochs",EPOCHS)
    mlflow.log_param("lr",0.01)

    train_loss_values = []
    test_loss_values = []
    epoch_count = []

    for epoch in range(EPOCHS):
        model_0.train()
        y_pred = model_0(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        model_0.eval()
        with torch.inference_mode():
            test_pred = model_0(X_test)
            test_loss = loss_fn(test_pred, y_test.type(torch.float))
            if epoch %10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
        mlflow.log_metrics({
            "train_loss":loss.item(),
            "test_loss":test_loss.item()
        },epoch)
    torch.save(model_0,'model.pt')
    mlflow.log_artifact('model.pt')