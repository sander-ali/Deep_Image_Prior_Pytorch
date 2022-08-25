import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from utils import network_arch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = cv2.resize(cv2.imread('test3.jpg'), (256, 256))
image_tensor = torch.Tensor(image).permute(2, 0, 1)/255
plt.imshow(image_tensor.permute(1, 2, 0))
plt.show()

net = network_arch()
# net = net.to(device)
x = torch.randn(1, 3, 256, 256)
print(net(x).shape)

iterations = 5000
criterion = nn.MSELoss()
lr = 1e-3
optimizer = optim.Adam(net.parameters(), lr=lr)

for i in tqdm(range(iterations)):
  out = net(x)
  loss = criterion(out, image_tensor.unsqueeze(0))
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if i % 100 == 0:
    print("Loss is {}".format(loss.item()))
    plt.subplot(1, 2, 1)
    pred_np = out.squeeze(0).permute(1, 2, 0).detach().numpy()
    plt.imshow(cv2.cvtColor(pred_np, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.close()