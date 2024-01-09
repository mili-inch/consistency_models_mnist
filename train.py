import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import wandb
from model.sampler import Sampler
from model.model import Model

wandb.init(
    project="mnist-consistency",
    config={
    "learning_rate": 0.001,
    "dataset": "MNIST",
    "epochs": 100,
    "batch_size": 512,
    "num_scales": 200,
    }
)

sampler = Sampler(num_scales=200)
model = Model()

transform = transforms.Compose(
    [transforms.ToTensor()]
)

trainset = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)
trainloader = th.utils.data.DataLoader(
    trainset,
    batch_size=512,
    shuffle=True,
    num_workers=2
)

testset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)
testloader = th.utils.data.DataLoader(
    testset, 
    batch_size=512,
    shuffle=False, 
    num_workers=2
)

model.train()
optim = optim.RAdam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.save(f'./output/epoch_{epoch}')
    wandb.log({"epoch": epoch})
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda() * 2 - 1
        labels = labels.cuda()
        loss = sampler.training_losses(model, inputs, labels) * 10
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()
        wandb.log({"loss": loss})
        print(loss.item())
    model.eval()
    with th.no_grad():
        sample = sampler.sample(model, 10, th.arange(10).cuda()) * 0.5 + 0.5
        wandb.log({"sample": [wandb.Image(x) for x in sample]})
