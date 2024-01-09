import torch as th
from model.sampler import Sampler
from model.model import Model

sampler = Sampler(num_scales=200)
model = Model()
model.load("./output/epoch_99")

model.eval()
with th.no_grad():
    sample = sampler.sample(model, 10, th.arange(10).cuda()) * 0.5 + 0.5
    import torchvision
    torchvision.utils.save_image(sample, "sample.png", nrow=10)