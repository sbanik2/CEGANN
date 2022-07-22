from math import *

import torch
import torch.nn as nn

from utilis import get_factor

# -------------------------------------------------------------------


class gbf_expansion(nn.Module):
    def __init__(self, gbf):
        super().__init__()
        self.min = gbf["dmin"]
        self.max = gbf["dmax"]
        self.steps = gbf["steps"]
        self.gamma = (self.max - self.min) / self.steps
        self.register_buffer(
            "filters", torch.linspace(self.min, self.max, self.steps)
        )

    def forward(self, data: torch.Tensor, bond=True) -> torch.Tensor:
        if bond:
            return torch.exp(
                -((data.unsqueeze(2) - self.filters) ** 2) / self.gamma**2
            )
        else:
            return torch.exp(
                -((data.unsqueeze(3) - self.filters) ** 2) / self.gamma**2
            )


# -------------------------------------------------------------------


class legendre_expansion(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.l = l

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        P0 = torch.clone(data)
        P0[:, :, :] = 1
        P1 = data
        if self.l == 0:
            return P0.unsqueeze(3) * get_factor(0)
        if self.l == 1:
            return torch.stack([P0, P1 * get_factor(1)], dim=3)
        else:
            factors = [get_factor(0), get_factor(1)]
            retvars = [P0, P1]
            for i in range(2, self.l):
                P = (1 / (i + 1)) * (
                    (2 * i + 1) * data * retvars[i - 1] - i * retvars[i - 2]
                )
                retvars.append(P)
                factors.append(get_factor(i))

        retvars = [var * factors[i] for i, var in enumerate(retvars)]
        return torch.stack(retvars, dim=3)


# -------------------------------------------------------------------


def Concat(atom_feature, bond_feature, nbr_idx):

    N, M = nbr_idx.shape
    _, O = atom_feature.shape
    _, _, P = bond_feature.shape

    index = nbr_idx.unsqueeze(1).expand(N, M, M)
    xk = atom_feature[index, :]
    xj = atom_feature[nbr_idx, :]
    xi = atom_feature.unsqueeze(1).expand(N, M, O)
    xij = torch.cat([xi, xj], dim=2)
    xij = xij.unsqueeze(2).expand(N, M, M, 2 * O)
    xijk = torch.cat([xij, xk], dim=3)

    eij = bond_feature.unsqueeze(2).expand(N, M, M, P)
    eik = bond_feature[nbr_idx, :]
    eijk = torch.cat([eij, eik], dim=3)

    return torch.cat([xijk, eijk], dim=3)


# -------------------------------------------------------------------


class ConvAngle(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(
        self,
        edge_fea_len,
        angle_fea_len,
    ):

        super(ConvAngle, self).__init__()

        self.angle_fea_len = angle_fea_len
        self.edge_fea_len = edge_fea_len

        # -------------------Angle----------------------

        self.lin_angle = nn.Linear(
            self.angle_fea_len + 2 * self.edge_fea_len, self.angle_fea_len
        )

        # ---------------Angle 3body attention------------------------

        self.attention_1 = nn.Linear(
            self.angle_fea_len + 2 * self.edge_fea_len, 1
        )
        self.leakyrelu_1 = nn.LeakyReLU(negative_slope=0.01)
        self.bn_ijkl = nn.LayerNorm(self.angle_fea_len + 2 * self.edge_fea_len)
        self.bn_ijkl = nn.LayerNorm(self.angle_fea_len)
        self.softplus_1 = nn.Softplus()

        self.bn_2 = nn.LayerNorm(self.angle_fea_len)
        self.softplus_2 = nn.Softplus()

    def forward(self, angle_fea, edge_fea, nbr_idx):

        N, M, O, P = angle_fea.shape

        # ---------------Edge update--------------------------

        eij = edge_fea.unsqueeze(2).expand(N, M, M, P)
        eik = edge_fea[nbr_idx, :]
        eijk = torch.cat([eij, eik], dim=3)

        angle_fea_cat = torch.cat([eijk, angle_fea], dim=3)

        attention_1 = self.attention_1(angle_fea_cat)
        alpha_1 = self.leakyrelu_1(attention_1)
        angle_fea_cat = alpha_1 * self.lin_angle(angle_fea_cat)

        angle_fea_summed = angle_fea_cat
        angle_fea_summed = angle_fea + angle_fea_summed
        angle_fea_summed = self.bn_2(angle_fea_summed)
        angle_fea_summed = self.softplus_2(angle_fea_summed)

        return angle_fea_summed


# -------------------------------------------------------------------


class ConvEdge(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, edge_fea_len, angle_fea_len):

        super(ConvEdge, self).__init__()

        self.edge_fea_len = edge_fea_len
        self.angle_fea_len = angle_fea_len

        # -------------------Angle----------------------
        self.lin_edge = nn.Linear(
            2 * self.edge_fea_len + self.angle_fea_len, self.edge_fea_len
        )

        # ---------------edege attention------------------------

        self.attention_1 = nn.Linear(
            2 * self.edge_fea_len + self.angle_fea_len, 1
        )
        self.leakyrelu_1 = nn.LeakyReLU(negative_slope=0.01)
        self.softmax_1 = nn.Softmax(dim=2)
        self.bn_1 = nn.LayerNorm(self.edge_fea_len)
        self.softplus_1 = nn.Softplus()

        self.bn_2 = nn.LayerNorm(self.edge_fea_len)
        self.softplus_2 = nn.Softplus()

    def forward(self, edge_fea, angle_fea, nbr_idx):

        N, M = nbr_idx.shape

        eij = edge_fea.unsqueeze(2).expand(N, M, M, self.edge_fea_len)

        eik = edge_fea[nbr_idx, :]

        edge_fea_cat = torch.cat([eij, eik, angle_fea], dim=3)

        attention_1 = self.attention_1(edge_fea_cat)
        alpha_1 = self.softmax_1(self.leakyrelu_1(attention_1))

        edge_fea_cat = alpha_1 * self.lin_edge(edge_fea_cat)

        edge_fea_cat = self.bn_1(edge_fea_cat)
        edge_fea_cat = self.softplus_1(edge_fea_cat)

        edge_fea_summed = edge_fea + torch.sum(edge_fea_cat, dim=2)

        edge_fea_summed = self.bn_2(edge_fea_summed)
        edge_fea_summed = self.softplus_2(edge_fea_summed)

        return edge_fea_summed


# -------------------------------------------------------------------


class CEGAN(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total material properties.
    """

    def __init__(
        self,
        gbf_bond,
        gbf_angle,
        n_conv_edge=3,
        h_fea_edge=128,
        h_fea_angle=128,
        n_classification=2,
        pooling=False,
        embedding=False,
    ):

        super(CEGAN, self).__init__()

        self.bond_fea_len = gbf_bond["steps"]
        self.angle_fea_len = gbf_angle["steps"]
        self.gbf_bond = gbf_expansion(gbf_bond)
        self.gbf_angle = gbf_expansion(gbf_angle)
        self.pooling = pooling
        self.embedding = embedding

        self.EdgeConv = nn.ModuleList(
            [
                ConvEdge(self.bond_fea_len, self.angle_fea_len)
                for _ in range(n_conv_edge)
            ]
        )
        self.AngConv = nn.ModuleList(
            [
                ConvAngle(self.bond_fea_len, self.angle_fea_len)
                for _ in range(n_conv_edge - 1)
            ]
        )

        self.expandEdge = nn.Linear(self.bond_fea_len, h_fea_edge)
        self.expandAngle = nn.Linear(self.angle_fea_len, h_fea_angle)

        self.bn = nn.LayerNorm(h_fea_edge + h_fea_angle)
        self.conv_to_fc_softplus = nn.Softplus()

        self.out = nn.Linear(h_fea_edge + h_fea_angle, n_classification)

        self.dropout = nn.Dropout()

    def forward(self, data):

        bond_fea, angle_fea, nbr_idx, crys_idx = data

        edge_fea = self.gbf_bond(bond_fea)
        angle_fea = self.gbf_angle(angle_fea, bond=False)

        # print(angle_fea.shape)

        edge_fea = self.EdgeConv[0](edge_fea, angle_fea, nbr_idx)

        for conv_edge, conv_angle in zip(self.EdgeConv[1:], self.AngConv):
            angle_fea = conv_angle(angle_fea, edge_fea, nbr_idx)
            edge_fea = conv_edge(edge_fea, angle_fea, nbr_idx)

        edge_fea = self.expandEdge(self.dropout(edge_fea))
        angle_fea = self.expandAngle(self.dropout(angle_fea))

        edge_fea = torch.sum(self.conv_to_fc_softplus(edge_fea), dim=1)

        # print(edge_fea.shape)

        angle_fea = torch.sum(
            self.conv_to_fc_softplus(
                torch.sum(self.conv_to_fc_softplus(angle_fea), dim=2)
            ),
            dim=1,
        )

        # print(angle_fea.shape)

        crys_fea = torch.cat([edge_fea, angle_fea], dim=1)
        # print(crys_fea.shape)

        if self.pooling:
            crys_fea = self.pool(crys_fea, crys_idx)
            # print("pooled",crys_fea.shape)

        crys_fea = self.conv_to_fc_softplus(self.bn(crys_fea))
        if self.embedding:
            embed = crys_fea

        crys_fea = self.dropout(crys_fea)
        out = self.out(crys_fea)
        # print(out.shape)

        if self.embedding:
            return out, embed

        else:
            return out

    def pool(self, atom_fea, crys_idx):

        # print(crystal_atom_idx)

        summed_fea = [
            torch.mean(
                atom_fea[idx_map[0] : idx_map[1], :], dim=0, keepdim=True
            )
            for idx_map in crys_idx
        ]
        return torch.cat(summed_fea, dim=0)
