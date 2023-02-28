import numpy as np
import torch
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import math


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal(param)

        # Initialize biases for LSTM’s forget gate to 1 to remember more by default. Similarly, initialize biases for GRU’s reset gate to -1.
        for names in m._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(m, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    elif classname.find('GRU') != -1:
        for name, param in m.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant(param, -1.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal(param)


def initial_model_weight(layers):
    for layer in layers:
        if list(layer.children()) == []:
            weights_init(layer)
            # print('weight initial finished!')
        else:
            for sub_layer in list(layer.children()):
                initial_model_weight([sub_layer])


class LayerNormGRUCell(torch.nn.Module):
    """
    https://github.com/ElektrischesSchaf/LayerNorm_GRU
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()

        self.ln_i2h = torch.nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_h2h = torch.nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_cell_1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_cell_2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.i2h = torch.nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.h_hat_W = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.h_hat_U = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):

        h = h
        h = h.view(h.size(0), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)

        # Layer norm
        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        # activations
        gates = preact[:, :].sigmoid()
        z_t = gates[:, :self.hidden_size]
        r_t = gates[:, -self.hidden_size:]

        # h_hat
        h_hat_first_half = self.h_hat_W(x)
        h_hat_last_half = self.h_hat_U(h)

        # layer norm
        h_hat_first_half = self.ln_cell_1( h_hat_first_half )
        h_hat_last_half = self.ln_cell_2( h_hat_last_half )

        h_hat = torch.tanh(  h_hat_first_half + torch.mul(r_t,   h_hat_last_half ) )

        h_t = torch.mul( 1-z_t , h ) + torch.mul( z_t, h_hat)

        # Reshape for compatibility

        h_t = h_t.view( h_t.size(0), -1)
        return h_t