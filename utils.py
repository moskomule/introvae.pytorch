import torch
from homura.utils import Trainer as TrainerBase
from torch import nn
from torch.nn import functional as F


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


def l_reg(mu, log_var):
    return - 0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=-1)


class Trainer(TrainerBase):
    def __init__(self, encoder, decoder, optimizer, callbacks, scheduler, hp, verb=False):
        super(Trainer, self).__init__(model=nn.ModuleList([encoder, decoder]), optimizer=optimizer, loss_f=None,
                                      callbacks=callbacks, scheduler=scheduler, verb=verb, hp=hp)
        self.encoder = self.model[0]
        self.decoder = self.model[1]

    def iteration(self, data, is_train):
        alpha = self.hp.alpha
        beta = self.hp.beta
        margin = self.hp.margin

        input, _ = self.to_device(data)
        if is_train:
            z_mu, z_log_var = self.encoder(input)
            z = reparameterize(z_mu, z_log_var)
            z_p = torch.randn_like(z)
            x_r = self.decoder(z)
            x_p = self.decoder(z_p)

            self.optimizer[0].zero_grad()
            z_r = self.encoder(x_r.detach())
            z_pp = self.encoder(x_p.detach())
            l_enc = l_reg(z_mu, z_log_var) + alpha * (F.relu(margin - l_reg(*z_r)) + F.relu(margin - l_reg(*z_pp)))
            (l_enc + beta * F.mse_loss(x_r, input)).backward()
            self.optimizer[0].step()

            self.optimizer[1].zero_grad()
            z_r = self.encoder(x_r)
            z_pp = self.encoder(x_p)
            l_dec = alpha * (l_reg(*z_r) + l_reg(*z_pp))
            (l_dec + beta * F.mse_loss(x_r, input)).backward()
            self.optimizer[1].step()

    def test(self, data_loader, name=None):
        pass

    def generate(self):
        pass
