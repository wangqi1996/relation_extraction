import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMEncoder(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 batch_first,
                 bidirectional):
        super(LSTMEncoder, self).__init__()
        input_size = input_size
        dropout = 0 if num_layers == 1 else dropout
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout, batch_first=batch_first,
                           bidirectional=bidirectional)

    def forward(self, inputs, lengths, need_sort=False):
        if need_sort:
            lengths, perm_idx = lengths.sort(0, descending=True)
            inputs = inputs[perm_idx]
        bsize = inputs.size()[0]
        # state_shape = self.config.n_cells,bsize,self.config.d_hidden
        # h0 = c0 = inputs.new_zeros(state_shape)
        inputs = pack(inputs, lengths, batch_first=True)
        outputs, (ht, ct) = self.rnn(inputs)
        outputs, _ = unpack(outputs, batch_first=True)

        if need_sort:
            _, unperm_idx = perm_idx.sort(0)
            outputs = outputs[unperm_idx]
        return outputs, ht.permute(1, 0, 2).contiguous().view(bsize, -1)


def mean_pool(input, length):
    # input: bsize *  seq_len * dim
    # length: bsize
    length = length.unsqueeze(1)
    input = torch.sum(input, 1).squeeze()
    return torch.div(input, length.float())


def max_pool(input, input_mask):
    # 1 -> 0, 0 -> -1e9
    input[input == 0] = -1e9

    result = torch.max(input, dim=1)
    return result[0]


def max_pool2(input, input_mask):
    # 1 -> 0, 0 -> -1e9
    input[input == 0] = -1e9

    result = torch.max(input, dim=1)
    return result[0], result[1]


class GateNetwork(nn.Module):
    def __init__(self, hidden_dim):
        super(GateNetwork, self).__init__()
        self.gate_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.gate_fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.gate_fc1.weight.data)
        nn.init.xavier_normal_(self.gate_fc2.weight.data)

    def forward(self, input1, input2):
        assert input1.size() == input2.size()
        gate = torch.sigmoid(
            self.gate_fc1(input1) +
            self.gate_fc2(input2)
        )
        return torch.mul(gate, input1) + torch.mul(1 - gate, input2)


def cummax(input, dim=-1):
    return torch.cumsum(torch.nn.functional.softmax(input, dim), dim=-1)


class ONLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, chunk_size):
        super(ONLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.n_chunk = int(hidden_size / chunk_size)

        self.ih = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_size + self.n_chunk * 2, bias=True),
            # LayerNorm(3 * hidden_size)
        )
        self.hh = nn.Linear(hidden_size, hidden_size * 4 + self.n_chunk * 2, bias=True)

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = self.ih(input) + self.hh(hx)
        cingate, cforgetgate = gates[:, :self.n_chunk * 2].chunk(2, 1)
        outgate, cell, ingate, forgetgate = gates[:, self.n_chunk * 2:].view(-1, self.n_chunk * 4,
                                                                             self.chunk_size).chunk(4, 1)

        cingate = 1. - cummax(cingate)
        cforgetgate = cummax(cforgetgate)

        cingate = cingate[:, :, None]
        cforgetgate = cforgetgate[:, :, None]

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cell = torch.tanh(cell)
        outgate = torch.sigmoid(outgate)

        overlap = cforgetgate * cingate
        forgetgate = forgetgate * overlap + (cforgetgate - overlap)
        ingate = ingate * overlap + (cingate - overlap)
        cy = forgetgate * cx + ingate * cell

        hy = outgate * torch.tanh(cy)
        return hy.view(-1, self.hidden_size), cy

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz, self.hidden_size).zero_().to(device),
                weight.new(bsz, self.n_chunk, self.chunk_size).zero_().to(device))

    def sample_masks(self):
        for m in self.drop_weight_modules:
            m.sample_mask()


class ONLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, chunk_size):
        super(ONLSTM, self).__init__()
        self.cell = ONLSTMCell(input_size, hidden_size, chunk_size)

    def init_hidden(self, bsz):
        return self.cell.init_hidden(bsz)

    def forward(self, input):
        batch_size, length, _ = input.size()
        input = input.permute(1, 0, 2)
        hx, cx = self.init_hidden(batch_size)
        outputs = []
        for x in input:
            hx, cx = self.cell(x, (hx, cx))
            outputs.append(hx)
        outputs = torch.stack(outputs, dim=1)
        return outputs, hx


if __name__ == '__main__':
    onlstm = ONLSTM(5, 10, 2)
    input = torch.randn(2, 3, 5)
    print(onlstm.forward(input)[0].size())
