import torch
import matplotlib.pyplot as plt

x_data=[1,2,3]
y_data=[2,4,6]

w = torch.tensor(1.0)
w.requires_grad = True
def forward(x):
    return x*w
def loss(x,y):
    y_pred = forward(x)
    return y_pred**2
w_list=[]
epoch_list=[]
print('predict(before training)',4,forward(4).item())
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l=loss(x,y)
        l.backward()
        print('\tgrad:',x,y,w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()
    w_list.append(w.data)
    epoch_list.append(epoch)
    print('progress:',epoch,l.item())
print('predict(after training):',4,forward(4).item())
plt.plot(epoch_list,w_list)
plt.show()