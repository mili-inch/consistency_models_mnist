import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from model.sampler import Sampler
from model.model import Model

sampler = Sampler(num_scales=40)
model = Model()
model.load("./output_0/epoch_42")

transform = transforms.Compose(
    [transforms.ToTensor()]
)

testset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)
testloader = th.utils.data.DataLoader(
    testset, 
    batch_size=1,
    shuffle=False, 
    num_workers=2
)

with th.no_grad():
    for i, data in enumerate(testloader, 0):
        inputs, labels = data

        inputs = inputs.cuda() * 2 - 1
        inputs = inputs.repeat(400, 1, 1, 1)
        labels = labels.cuda()
        print(labels)
        noise = th.randn(400, 1, 28, 28, device="cuda")
        conds = th.arange(400, device="cuda") // 40
        indices = th.arange(40, device="cuda").repeat(10).flatten()
        t = sampler.get_t(indices)

        x_t = inputs + noise * t[..., None, None, None]
        x_0 = sampler.denoise(model, x_t, t, conds)
        norm = th.abs(((x_t - x_0) / t[..., None, None, None] - noise))
        norm = norm.mean(dim=(1, 2, 3), keepdim=False)
        norm = norm.reshape(10, 40)
        norm = norm.mean(dim=1, keepdim=False)
        print(norm.argmax())