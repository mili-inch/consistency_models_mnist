import torch as th
import torch.nn.functional as F
from model.unet import UNet
from model.embeddings import TimestepEmbedding, ClassEmbedding

class Model():
    def __init__(self):
        self.unet = UNet(1, 256, 256).cuda()
        self.time_embed = TimestepEmbedding(256).cuda()
        self.cond_embed = ClassEmbedding(10, 256).cuda()

    def __call__(self, scaled_input, t, conds):
        t = t.unsqueeze(-1)
        c = F.one_hot(conds, 10).float()
        time_embedded = self.time_embed(t)
        cond_embedded = self.cond_embed(c)

        output = self.unet(scaled_input, time_embedded, cond_embedded)
        return output
    
    def save(self, path):
        th.save(self.unet.state_dict(), path)
        th.save(self.time_embed.state_dict(), path + "_time_embed")
        th.save(self.cond_embed.state_dict(), path + "_cond_embed")

    def load(self, path):
        self.unet.load_state_dict(th.load(path))
        self.time_embed.load_state_dict(th.load(path + "_time_embed"))
        self.cond_embed.load_state_dict(th.load(path + "_cond_embed"))

    def train(self):
        self.unet.train()
        self.time_embed.train()
        self.cond_embed.train()

    def eval(self):
        self.unet.eval()
        self.time_embed.eval()
        self.cond_embed.eval()

    def parameters(self):
        return list(self.unet.parameters()) + list(self.time_embed.parameters()) + list(self.cond_embed.parameters())