import torch
import torch.nn as nn
import torch.nn.functional as F
import util



class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,wv->ncwl', (x, A))
        return x.contiguous()


class d_nconv(nn.Module):
    def __init__(self):
        super(d_nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,nvw->ncvl', (x, A))
        return x.contiguous()




class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class linear_(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear_, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 2), dilation=2, padding=(0, 0), stride=(1, 1),
                                   bias=True)

    def forward(self, x):
        return self.mlp(x)




class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class dgcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=2):
        super(dgcn, self).__init__()
        self.d_nconv = d_nconv()
        c_in = (order * 3 + 1) * c_in
        self.mlp = linear_(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.d_nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.d_nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h




class hgcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=2):
        super(hgcn, self).__init__()
        self.nconv = nconv()
        c_in = (order + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, G):
        out = [x]
        support = [G]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class hgcn_edge_At(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=1):
        super(hgcn_edge_At, self).__init__()
        self.nconv = nconv()
        c_in = (order + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, G):
        out = [x]
        support = [G]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class dhgcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=2):
        super(dhgcn, self).__init__()
        self.d_nconv = d_nconv()
        c_in = (order + 1) * c_in
        self.mlp = linear_(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, G):
        out = [x]
        support = [G]
        for a in support:
            x1 = self.d_nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.d_nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h




class spatial_attention(nn.Module):
    def __init__(self, in_channels, num_of_timesteps, num_of_edge, num_of_vertices):
        super(spatial_attention, self).__init__()
        self.W1 = nn.Parameter(torch.randn(num_of_timesteps).cuda(), requires_grad=True).cuda()
        self.W2 = nn.Parameter(torch.randn(num_of_timesteps).cuda(), requires_grad=True).cuda()
        self.W3 = nn.Parameter(torch.randn(in_channels,int(in_channels/2)).cuda(), requires_grad=True).cuda()
        self.W4 = nn.Parameter(torch.randn(in_channels,int(in_channels/2)).cuda(), requires_grad=True).cuda()
        self.out_conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=in_channels,
                                  kernel_size=(1, 1))

    def forward(self, x, idx, idy):
        lhs = torch.matmul(torch.matmul(x, self.W1),self.W3)
        rhs = torch.matmul(torch.matmul(x, self.W2),self.W4)
        sum = torch.cat([lhs[:,idx,:], rhs[:,idy,:]], dim=2)
        sum = torch.unsqueeze(sum, dim=3).transpose(1, 2)
        S = self.out_conv(sum)
        S = torch.squeeze(S).transpose(1, 2)
        return S






class ddstgcn(nn.Module):
    def __init__(self, batch_size, H_a, H_b, G0, G1, indices, G0_all, G1_all, H_T_new, lwjl,num_nodes,
                 dropout=0.3, supports=None, in_dim=2, out_dim=12, residual_channels=40, dilation_channels=40,
                 skip_channels=320, end_channels=640, kernel_size=2, blocks=3, layers=1):
        super(ddstgcn, self).__init__()
        self.batch_size = batch_size
        self.H_a = H_a
        self.H_b = H_b
        self.G0 = G0
        self.G1 = G1
        self.H_T_new = H_T_new
        self.lwjl = lwjl
        self.indices = indices
        self.G0_all = G0_all
        self.G1_all = G1_all

        self.edge_node_vec1 = nn.Parameter(torch.rand(self.H_a.size(1), 10).cuda(), requires_grad=True).cuda()
        self.edge_node_vec2 = nn.Parameter(torch.rand(10, self.H_a.size(0)).cuda(), requires_grad=True).cuda()

        self.node_edge_vec1 = nn.Parameter(torch.rand(self.H_a.size(0), 10).cuda(), requires_grad=True).cuda()
        self.node_edge_vec2 = nn.Parameter(torch.rand(10, self.H_a.size(1)).cuda(), requires_grad=True).cuda()

        self.hgcn_w_vec_edge_At_forward = nn.Parameter(torch.rand(self.H_a.size(1)).cuda(),
                                                       requires_grad=True).cuda()
        self.hgcn_w_vec_edge_At_backward = nn.Parameter(torch.rand(self.H_a.size(1)).cuda(),
                                                        requires_grad=True).cuda()

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.dgconv = nn.ModuleList()
        self.filter_convs_h = nn.ModuleList()
        self.gate_convs_h = nn.ModuleList()
        self.SAt_forward = nn.ModuleList()
        self.SAt_backward = nn.ModuleList()
        self.hgconv_edge_At_forward = nn.ModuleList()
        self.hgconv_edge_At_backward = nn.ModuleList()
        self.gconv_dgcn_w = nn.ModuleList()
        self.dhgconv = nn.ModuleList()
        self.bn_g = nn.ModuleList()
        self.bn_hg = nn.ModuleList()


        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports
        self.num_nodes = num_nodes
        receptive_field = 1
        self.supports_len = 0
        self.supports_len += len(supports)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).cuda(), requires_grad=True).cuda()
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).cuda(), requires_grad=True).cuda()
        self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size
            new_dilation = 2
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.filter_convs_h.append(nn.Conv2d(in_channels=1+residual_channels*2,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs_h.append(nn.Conv2d(in_channels=1+residual_channels*2,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.SAt_forward.append(spatial_attention( residual_channels, int(13-receptive_field+1),
                                                                 self.indices.size(1), num_nodes))
                self.SAt_backward.append(spatial_attention( residual_channels, int(13-receptive_field+1),
                                                                  self.indices.size(1), num_nodes))
                receptive_field += (additional_scope * 2)
                self.dgconv.append(dgcn(dilation_channels, int(residual_channels / 2), dropout))
                self.hgconv_edge_At_forward.append(hgcn_edge_At(residual_channels, 1, dropout))
                self.hgconv_edge_At_backward.append(hgcn_edge_At(residual_channels, 1, dropout))
                self.gconv_dgcn_w.append(
                    gcn((residual_channels), 1, dropout, support_len=2, order=1))
                self.dhgconv.append(dhgcn(dilation_channels, int(residual_channels / 2), dropout))
                self.bn_g.append(nn.BatchNorm2d(int(residual_channels / 2)))
                self.bn_hg.append(nn.BatchNorm2d(int(residual_channels / 2)))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_start = nn.BatchNorm2d(in_dim, affine=False)
        self.new_supports_w = [
            torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]).cuda()]
        self.new_supports_w = self.new_supports_w + [
            torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]).cuda()]
        self.new_supports_w = self.new_supports_w + [
            torch.zeros([self.num_nodes, self.num_nodes]).repeat([self.batch_size, 1, 1]).cuda()]


    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.bn_start(x)
        x = self.start_conv(x)
        skip = 0
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        adp_new = adp.repeat([self.batch_size, 1, 1])
        new_supports = self.supports + [adp]
        edge_node_H = (self.H_T_new * (torch.mm(self.edge_node_vec1, self.edge_node_vec2)))
        self.H_a_ = (self.H_a * (torch.mm(self.node_edge_vec1, self.node_edge_vec2)))
        self.H_b_ = (self.H_b * (torch.mm(self.node_edge_vec1, self.node_edge_vec2)))
        G0G1_edge_At_forward = self.G0_all @ (torch.diag_embed((self.hgcn_w_vec_edge_At_forward))) @ self.G1_all
        G0G1_edge_At_backward = self.G0_all @ (torch.diag_embed((self.hgcn_w_vec_edge_At_backward))) @ self.G1_all
        self.new_supports_w[2] = adp_new.cuda()
        forward_medium = torch.eye(self.num_nodes).repeat([self.batch_size, 1, 1]).cuda()
        backward_medium = torch.eye(self.num_nodes).repeat([self.batch_size, 1, 1]).cuda()

        for i in range(self.blocks * self.layers):
            edge_feature = util.feature_node_to_edge(x, self.H_a_, self.H_b_, operation="concat")
            edge_feature = torch.cat([edge_feature, self.lwjl.repeat(1, 1, 1, edge_feature.size(3))], dim=1)
            filter_h = self.filter_convs_h[i](edge_feature)
            filter_h = torch.tanh(filter_h)
            gate_h = self.gate_convs_h[i](edge_feature)
            gate_h = torch.sigmoid(gate_h)
            x_h = filter_h * gate_h

            batch_edge_forward = self.SAt_forward[i](x.transpose(1,2),self.indices[0],self.indices[1])
            batch_edge_backward = self.SAt_backward[i](x.transpose(1,2),self.indices[0],self.indices[1])
            batch_edge_forward = torch.unsqueeze(batch_edge_forward, dim=3)
            batch_edge_forward = batch_edge_forward.transpose(1,2)
            batch_edge_forward = self.hgconv_edge_At_forward[i](batch_edge_forward, G0G1_edge_At_forward)
            batch_edge_forward = torch.squeeze(batch_edge_forward)
            forward_medium[:,self.indices[0],self.indices[1]] = torch.sigmoid((batch_edge_forward))
            self.new_supports_w[0] = forward_medium
            batch_edge_backward = torch.unsqueeze(batch_edge_backward, dim=3)
            batch_edge_backward = batch_edge_backward.transpose(1, 2)
            batch_edge_backward = self.hgconv_edge_At_backward[i](batch_edge_backward, G0G1_edge_At_backward)
            batch_edge_backward = torch.squeeze(batch_edge_backward)
            backward_medium[:,self.indices[0],self.indices[1]] = torch.sigmoid((batch_edge_backward))
            self.new_supports_w[1] = backward_medium.transpose(1,2)
            self.new_supports_w[0] = self.new_supports_w[0] * new_supports[0]
            self.new_supports_w[1] = self.new_supports_w[1] * new_supports[1]
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = self.dgconv[i](x, self.new_supports_w)
            x = self.bn_g[i](x)

            dhgcn_w_input = residual
            dhgcn_w_input = dhgcn_w_input.transpose(1, 2)
            dhgcn_w_input = torch.mean(dhgcn_w_input, 3)
            dhgcn_w_input = dhgcn_w_input.transpose(0, 2)
            dhgcn_w_input = torch.unsqueeze(dhgcn_w_input, dim=0)
            dhgcn_w_input = self.gconv_dgcn_w[i](dhgcn_w_input, self.supports)
            dhgcn_w_input = torch.squeeze(dhgcn_w_input)
            dhgcn_w_input = dhgcn_w_input.transpose(0, 1)
            dhgcn_w_input = self.G0 @ (torch.diag_embed(dhgcn_w_input)) @ self.G1
            x_h = self.dhgconv[i](x_h, dhgcn_w_input)
            x_h = self.bn_hg[i](x_h)

            x = util.fusion_edge_node(x, x_h, edge_node_H)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

        x = F.leaky_relu(skip)
        x = F.leaky_relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x


