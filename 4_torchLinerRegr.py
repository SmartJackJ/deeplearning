import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):   #初始化设置
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1) #线性模型实例化 y=xw^t+b 后面的输入是特征的维数，即矩阵的列数

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

criterion = torch.nn.MSELoss(False)   ##对损失函数类实例化
optimizer = torch.optim.SGD(model.parameters(), lr=0.05) ##对优化模型类实例化

for epoch in range(1000):
    y_pred = model(x_data)       ##计算y预测
    loss = criterion(y_pred, y_data) ##计算损失函数，并生成图
    print(epoch, loss.item())

    optimizer.zero_grad()  ##梯度归零
    loss.backward()    ##反向传播计算梯度
    optimizer.step()   ##自动更新

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)