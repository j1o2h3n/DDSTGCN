import torch.optim as optim
from model import *
import util

class trainer():
    def __init__(self, batch_size, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay,
                 supports, H_a, H_b, G0, G1,indices,G0_all,G1_all, H_T_new, lwjl, clip=3, lr_de_rate=0.97):
        self.model = ddstgcn( batch_size, H_a, H_b, G0,G1, indices, G0_all,G1_all, H_T_new,
                              lwjl, num_nodes, dropout, supports=supports, in_dim=in_dim,
                              out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid,
                              skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = clip
        lr_decay_rate=lr_de_rate
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
