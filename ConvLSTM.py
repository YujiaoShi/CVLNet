import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                      out_channels=4 * self.hidden_dim,
                      kernel_size=self.kernel_size,
                      padding=self.padding,
                      bias=self.bias),
            nn.BatchNorm2d(4*self.hidden_dim)
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_dim,
                     out_channels=self.out_dim,
                     kernel_size=self.kernel_size,
                     padding=self.padding,
                     bias=self.bias),
            nn.BatchNorm2d(self.out_dim)
        )

    def forward(self, input_tensor, h_state, c_state):

        combined = torch.cat([input_tensor, h_state], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_state + i * g     # [B, hidden, H, W]
        h_next = o * torch.tanh(c_next)  # [B, hidden, H, W]

        output = self.outconv(h_next)    # [B, out_dim, H, W]

        return output, h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv[0].weight.device)


class StackedConvLSTM(nn.Module):
    """
    Parameters:
        input_dim:
            Number of channels in input
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size:
            Size of kernel in convolutions
        num_layers:
            Number of LSTM layers stacked on each other
        bias:
            Bias or no bias in Convolution
        output_dim:
            the final output dimension of the stacked (last) LSTM layers
        Note: Will do same padding.
    Input:
        A tensor of size [B, C, H, W]
    Output:
        y

    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim,
                 bias=True):
        super(StackedConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        # kernel size should be a tuple

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim     # list, for all layers
        self.kernel_size = kernel_size   # list, for all layers
        self.num_layers = num_layers
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cur_output_dim = self.hidden_dim[i] if i < self.num_layers -1 else self.output_dim

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          out_dim=cur_output_dim,
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        # when to use nn.ModuleList and when to use nn.Sequential

    def forward(self, input_tensor, c_state=None, h_state=None):
        """
        Parameters
        ----------
        input_tensor:
            4-D Tensor of shape [B, C, H, W]
        c_state:
            either None or a list, each item in the list has a shape of [B, hidden_dim_in, H, W].
        h_list:
            either None or a list, each item in the list has a shape of [B, hidden_dim_in, H, W].
        Returns
        -------
        last_state_list, layer_output
        """
        B, C, H, W = input_tensor.size()

        # Implement stateful ConvLSTM
        if c_state is None:
            c_state = self._init_hidden(batch_size=B, image_size=(H, W))

        if h_state is None:
            h_state = self._init_hidden(batch_size=B, image_size=(H, W))

        y = input_tensor

        h_state_output = []
        c_state_output = []

        for layer_idx in range(self.num_layers):

            h_in = h_state[layer_idx]
            c_in = c_state[layer_idx]

            y, h_out, c_out = self.cell_list[layer_idx](y, h_in, c_in)

            h_state_output.append(h_out)
            c_state_output.append(c_out)

        return y, h_state_output, c_state_output

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class VE_LSTM3D(nn.Module):
    """
    Visbility and Elevation probability estimation, using LSTM3D
    Parameters:
        input_dim: int
            Number of channels in input
        hidden_dim: int
            Number of hidden channels
        kernel_size: tuple
            Size of kernel in convolutions
        num_layers: int
            Number of LSTM layers stacked on each other
        bias: bool
            Bias or no bias in Convolution
        order: int
            0 for forward, 1 for inverse, 2 for bilateral
        Note: Will do same padding.
    Input:
        A tensor of size [B, S, E, C, H, W]
    Output:

    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim,
                 bias=True, seq_order=0, ele_order=0, height_planes=8):
        super(VE_LSTM3D, self).__init__()

        self.input_dim = input_dim                # int
        self.output_dim = output_dim              # int
        self.hidden_dim = hidden_dim              # int

        self.kernel_size = kernel_size            # tuple
        self.num_layers = num_layers              # int
        self.bias = bias                          # bool
        self.seq_order = seq_order                # int
        self.ele_order = ele_order                # int

        self.LSTM_layer = StackedConvLSTM(input_dim=self.input_dim,
                                          hidden_dim=self.hidden_dim,
                                          kernel_size=self.kernel_size,
                                          num_layers=self.num_layers,
                                          output_dim=self.hidden_dim,
                                          bias=self.bias)

        channel_sum = self.hidden_dim

        if self.seq_order==2:
            self.LSTM_layer_seqInv = StackedConvLSTM(input_dim=self.input_dim,
                                                     hidden_dim=self.hidden_dim,
                                                     kernel_size=self.kernel_size,
                                                     num_layers=self.num_layers,
                                                     output_dim=self.hidden_dim,
                                                     bias=self.bias)
            channel_sum += self.hidden_dim

        if ele_order == 2:
            self.LSTM_layer_eleInv = StackedConvLSTM(input_dim=self.input_dim,
                                                     hidden_dim=self.hidden_dim,
                                                     kernel_size=self.kernel_size,
                                                     num_layers=self.num_layers,
                                                     output_dim=self.hidden_dim,
                                                     bias=self.bias)
            channel_sum += self.hidden_dim

            if seq_order == 2:
                self.LSTM_layer_eleInv_seqInv = StackedConvLSTM(input_dim=self.input_dim,
                                                                hidden_dim=self.hidden_dim,
                                                                kernel_size=self.kernel_size,
                                                                num_layers=self.num_layers,
                                                                output_dim=self.hidden_dim,
                                                                bias=self.bias)
                channel_sum += self.hidden_dim

        self.convs = nn.Sequential(
            # nn.Conv2d(in_channels=channel_sum, out_channels=int(channel_sum//2), kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.BatchNorm2d(int(channel_sum//2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel_sum, out_channels=self.output_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(self.output_dim)
        )  # input [B*S*E, H, W, (4)C] --> output [B*S*E, H, W, self.output_dim]

        # self.LSTM_height = StackedConvLSTM(
        #     input_dim=self.output_dim - 1,
        #     hidden_dim=self.output_dim - 1,
        #     kernel_size=self.kernel_size,
        #     num_layers=self.num_layers,
        #     output_dim=self.out_channels - 1,
        #     bias=self.bias
        # )
        # channel_sum_2 = self.output_dim - 1
        # if ele_order == 2:
        #     self.LSTM_height_inv = StackedConvLSTM(
        #         input_dim=self.output_dim - 1,
        #         hidden_dim=self.output_dim - 1,
        #         kernel_size=self.kernel_size,
        #         num_layers=self.num_layers,
        #         output_dim=self.out_channels - 1,
        #         bias=self.bias
        #     )
        #     channel_sum_2 += self.output_dim - 1
        #
        # self.conv_height = nn.Sequential(
        #     nn.Conv2d(in_channels=channel_sum_2, out_channels=1, kernel_size=(3, 3),
        #               stride=(1, 1), padding=1),
        #     nn.BatchNorm2d(int(channel_sum // 2))
        # )

        self.conv_height = nn.Sequential(
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.output_dim - 1, out_channels=int((self.output_dim - 1)//2), kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.BatchNorm2d(int((self.output_dim - 1)//2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=(self.output_dim - 1)*height_planes, out_channels=height_planes, kernel_size=(3, 3), stride=(1, 1),
                      padding=1)
        )
        # self.conv_height[-1].weight = nn.Parameter(torch.zeros(height_planes, (self.ele_output_dim - 1) * height_planes, 3, 3))
        # self.conv_height[-1].bias = nn.Parameter(torch.cat([torch.zeros(height_planes - 1), torch.ones(1)], dim=0))

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor:
            6-D Tensor of shape [B, S, E, C, H, W]
        Returns
        -------

        """
        B, S, E, C, H, W = input_tensor.size()

        if self.seq_order==1:  # inverse the tensor along the sequence dimension
            input_tensor = input_tensor[:, ::-1, :, :, :, :]

        if self.ele_order==1:  # inverse the tensor along the elevation dimension
            input_tensor = input_tensor[:, :, ::-1, :, :, :]


        y_out = []
        for ele_idx in range(0, E):
            if ele_idx==0:
                c_state_seqfwd_list = [None]*S
                c_state_seqinv_list = [None]*S
            
            y_seq_list = []
            for seq_idx in range(0, S):
                if seq_idx==0: 
                    h_state_fwd = None
                    
                c_state_seqfwd_in = c_state_seqfwd_list[seq_idx]
                y_seqfwd, h_state_fwd, c_state_seqfwd_out = self.LSTM_layer(input_tensor[:, seq_idx, ele_idx, :, :, :],
                                                              h_state_fwd, c_state_seqfwd_in)
                y_seq_list.append(y_seqfwd)  # [
                c_state_seqfwd_list[seq_idx] = c_state_seqfwd_out
            y_seq_out = torch.stack(y_seq_list, dim=1)  # [B, S, C, H, W]

            if self.seq_order == 2:
                y_seqinv_list = []
                for seq_idx in reversed(range(0, S)):
                    if seq_idx==S-1:
                        h_state_inv = None

                    c_state_seqinv_in = c_state_seqinv_list[seq_idx]
                    y_seqinv, c_state_seqinv_out, h_state_inv = self.LSTM_layer_seqInv(input_tensor[:, seq_idx, ele_idx, :, :, :],
                                                              c_state_seqinv_in, h_state_inv)
                    y_seqinv_list.append(y_seqinv)
                    c_state_seqinv_list[seq_idx] = c_state_seqinv_out

                y_seqinv_out = torch.stack(y_seqinv_list, dim=1)  # [B, S, C, H, W]
                y_seq_out = torch.cat([y_seq_out, y_seqinv_out], dim=2)  # [B, S, 2C, H, W]

            y_out.append(y_seq_out)

        y_out = torch.stack(y_out, dim=2)  # [B, S, E, C(2C), H, W]

        if self.ele_order==2:
            y_eleinv_out = []
            for ele_idx in reversed(range(0, E)):
                if ele_idx == E-1:
                    c_state_seqfwd_list = [None] * S
                    c_state_seqinv_list = [None] * S

                y_seq_list = []
                for seq_idx in range(0, S):
                    if seq_idx == 0:
                        h_state_fwd = None

                    c_state_seqfwd_in = c_state_seqfwd_list[seq_idx]
                    y_seqfwd, h_state_fwd, c_state_seqfwd_out = self.LSTM_layer_eleInv(
                        input_tensor[:, seq_idx, ele_idx, :, :, :],
                        h_state_fwd, c_state_seqfwd_in)
                    y_seq_list.append(y_seqfwd)  # [
                    c_state_seqfwd_list[seq_idx] = c_state_seqfwd_out
                y_seq_out = torch.stack(y_seq_list, dim=1)  # [B, S, C, H, W]

                if self.seq_order == 2:
                    y_seqinv_list = []
                    for seq_idx in reversed(range(0, S)):
                        if seq_idx == S - 1:
                            h_state_inv = None

                        c_state_seqinv_in = c_state_seqinv_list[seq_idx]
                        y_seqinv, c_state_seqinv_out, h_state_inv = self.LSTM_layer_eleInv_seqInv(
                            input_tensor[:, seq_idx, ele_idx, :, :, :],
                            c_state_seqinv_in, h_state_inv)
                        y_seqinv_list.append(y_seqinv)
                        c_state_seqinv_list[seq_idx] = c_state_seqinv_out

                    y_seqinv_out = torch.stack(y_seqinv_list, dim=1)  # [B, S, C, H, W]
                    y_seq_out = torch.cat([y_seq_out, y_seqinv_out], dim=2)

                y_eleinv_out.append(y_seq_out)

            y_eleinv_out = torch.stack(y_eleinv_out, dim=2)  # [B, S, E, C, H, W]

            y_out = torch.cat([y_out, y_eleinv_out], dim=3)  # [B, S, E, C, H, W]

        y_out = y_out.view(B*S*E, -1, H, W)
        y_out = self.convs(y_out)

        vis, vis_feat = torch.split(y_out, [1, self.output_dim-1], dim=1)
        vis = nn.Softmax(dim=1)(vis.view(B, S, E, 1, H, W))
        vis_feat = vis_feat.view(B, S, E, self.output_dim-1, H, W)
        vis_feat = torch.mean(vis_feat, dim=1)  # [B, E, C, H, W]

        # y_out_list = []
        # for ele_idx in range(0, E):
        #     if ele_idx==0:
        #         h_state_fwd = None
        #         c_state_fwd = None
        #
        #     y, h_state_fwd, c_state_fwd = self.LSTM_height(vis_feat[:, ele_idx, :, :, :], h_state_fwd, c_state_fwd)
        #     y_out_list.append(y)
        # y_out = torch.stack(y_out_list, dim=1)  # [B, E, C, H, W]
        #
        # if self.ele_order == 2:
        #     y_out_eleinv_list = []
        #     for ele_idx in reversed(range(0, E)):
        #         if ele_idx==E-1:
        #             h_state_inv = None
        #             c_state_inv = None
        #         y, h_state_inv, c_state_inv = self.LSTM_height_inv(vis_feat[:, ele_idx, :, :, :], h_state_inv, c_state_inv)
        #         y_out_eleinv_list.append(y)
        #     y_out_eleinv = torch.stack(y_out_eleinv_list, dim=1) # [B, E, C, H, W]
        #     y_out = torch.cat([y_out, y_out_eleinv], dim=2)
        #
        # y_out = y_out.view(B*E, -1, H, W)
        # ele_prob = self.conv_height(y_out)
        # ele_prob = nn.Softmax(dim=1)(ele_prob.view(B, E, 1, H, W))

        vis_feat = vis_feat.view(B, -1, H, W)
        ele_feat = self.conv_height(vis_feat)  # [B, E, H, W]
        ele_prob = nn.Softmax(dim=1)(ele_feat.view(B, E, 1, H, W))

        return vis, ele_prob  # [B, S, E, 1, H, W], [B, E, 1, H, W]


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        B, S, C, H, W = x.shape
        return x.view(B*S, C, H, W)

class Reshape_SC(nn.Module):
    def __init__(self):
        super(Reshape_SC, self).__init__()

    def forward(self, x):
        B, S, C, H, W = x.shape
        return x.reshape(B, S*C, H, W)



class VE_LSTM2D(nn.Module):
    ''' LSTM 2D for visibility and elevation estimation'''
    def __init__(self, seq_input_dim, hidden_dim, kernel_size, num_layers, ele_output_dim, seq_output_dim,
                 bias=True, ele_order=0, seq_order=0, seq_fuse='LSTM', height_planes=8):
        # seq_fuse: conv or LSTM
        super(VE_LSTM2D, self).__init__()

        self.seq_input_dim = seq_input_dim  # int
        self.seq_output_dim = seq_output_dim  # int

        self.ele_input_dim = seq_output_dim
        self.ele_output_dim = ele_output_dim

        self.hidden_dim = hidden_dim  # int

        self.kernel_size = kernel_size  # tuple
        self.num_layers = num_layers  # int
        self.bias = bias  # bool
        self.ele_order = ele_order  # int

        if seq_fuse == 'conv':
            self.SeqNet = nn.Sequential(
                Reshape(),
                nn.Conv2d(in_channels=self.seq_input_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=self.seq_output_dim, kernel_size=(3, 3), stride=(1, 1), padding=1)
            )   # input [B, S, C, H, W] output [B*S, C, H, W]
        elif seq_fuse == 'LSTM':
            self.SeqNet = S_LSTM2D(self.seq_input_dim, hidden_dim, kernel_size, num_layers, self.seq_output_dim,
                 bias=True, order=seq_order)


        self.LSTM_layer = StackedConvLSTM(input_dim=self.ele_input_dim,
                                          hidden_dim=self.hidden_dim,
                                          kernel_size=self.kernel_size,
                                          num_layers=self.num_layers,
                                          output_dim=self.hidden_dim,
                                          bias=self.bias)

        channel_sum = self.hidden_dim

        if self.ele_order == 2:
            self.LSTM_layer_Inv = StackedConvLSTM(input_dim=self.ele_input_dim,
                                                     hidden_dim=self.hidden_dim,
                                                     kernel_size=self.kernel_size,
                                                     num_layers=self.num_layers,
                                                     output_dim=self.hidden_dim,
                                                     bias=self.bias)
            channel_sum += self.hidden_dim

        self.convs = nn.Sequential(
            # nn.Conv2d(in_channels=channel_sum, out_channels=int(channel_sum//2), kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.BatchNorm2d(int(channel_sum//2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel_sum, out_channels=self.ele_output_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(self.ele_output_dim)
        )  # input [B*S*E, H, W, (4)C] --> output [B*S*E, H, W, self.output_dim]

        self.conv_height = nn.Sequential(
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.output_dim - 1, out_channels=int((self.output_dim - 1)//2), kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.BatchNorm2d(int((self.output_dim - 1)//2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=(self.ele_output_dim - 1)*height_planes, out_channels=height_planes, kernel_size=(3, 3), stride=(1, 1),
                      padding=1)
        )
        # self.conv_height[-1].weight = nn.Parameter(torch.zeros(height_planes, (self.ele_output_dim - 1)*height_planes, 3, 3))
        # self.conv_height[-1].bias = nn.Parameter(torch.cat([torch.zeros(height_planes - 1), torch.ones(1)], dim=0))

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor:
            6-D Tensor of shape [B, S, E, C, H, W]
        Returns
        -------

        """
        B, S, E, C, H, W = input_tensor.size()

        if self.ele_order == 1:  # inverse the tensor along the elevation dimension
            input_tensor = input_tensor[:, :, ::-1, :, :, :]

        # ======== First encode relationship between sequences ================
        seq_out = self.SeqNet(input_tensor.transpose(1, 2).contiguous().view(B*E, S, C, H, W))  # [B*E*S, C, H, W]
        seq_out = seq_out.contiguous().view(B, E, S, C, H, W).transpose(1, 2)
        seq_out = seq_out.contiguous().view(B*S, E, C, H, W)

        y_out = []
        h_state_fwd = None
        c_state_fwd = None
        for ele_idx in range(0, E):

            y_fwd, h_state_fwd, c_state_fwd = self.LSTM_layer(seq_out[:, ele_idx, :, :, :], h_state_fwd, c_state_fwd)
            # [B*S, C, H, W]
            y_out.append(y_fwd)
        y_out = torch.stack(y_out, dim=1)  # [B*S, E, C H, W]

        if self.ele_order == 2:
            y_inv_out = []
            h_state_inv = None
            c_state_inv = None
            for ele_idx in reversed(range(0, E)):
                y_inv, h_state_inv, c_state_inv = self.LSTM_layer_Inv(seq_out[:, ele_idx, :, :, :],
                                                                      h_state_inv, c_state_inv)
                y_inv_out.append(y_inv)

            y_inv_out = torch.stack(y_inv_out, dim=1)
            y_out = torch.cat([y_out, y_inv_out], dim=2)

        y_out = y_out.view(B*S*E, -1, H, W)
        y_out = self.convs(y_out)

        vis, vis_feat = torch.split(y_out, [1, self.ele_output_dim - 1], dim=1)
        vis = nn.Softmax(dim=1)(vis.view(B, S, E, 1, H, W))
        vis_feat = vis_feat.view(B, S, E, self.ele_output_dim - 1, H, W)
        vis_feat = torch.mean(vis_feat, dim=1)  # [B, E, C, H, W]

        vis_feat = vis_feat.view(B, -1, H, W)
        ele_feat = self.conv_height(vis_feat)  # [B, E, H, W]
        ele_prob = nn.Softmax(dim=1)(ele_feat.view(B, E, 1, H, W))

        return vis, ele_prob  # [B, S, E, 1, H, W], [B, E, 1, H, W]


class S_LSTM2D(nn.Module):
    ''' LSTM 2D for visibility and elevation estimation'''

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, output_dim,
                 bias=True, order=0):
        # time_dim: Elevation or Sequence
        super(S_LSTM2D, self).__init__()

        self.input_dim = input_dim  # int
        self.output_dim = output_dim  # int
        self.hidden_dim = hidden_dim  # int

        self.kernel_size = kernel_size  # tuple
        self.num_layers = num_layers  # int
        self.bias = bias  # bool
        self.order = order  # int

        self.LSTM_layer = StackedConvLSTM(input_dim=self.input_dim,
                                          hidden_dim=self.hidden_dim,
                                          kernel_size=self.kernel_size,
                                          num_layers=self.num_layers,
                                          output_dim=self.hidden_dim,
                                          bias=self.bias)

        channel_sum = self.hidden_dim

        if self.order == 2:
            self.LSTM_layer_Inv = StackedConvLSTM(input_dim=self.input_dim,
                                                  hidden_dim=self.hidden_dim,
                                                  kernel_size=self.kernel_size,
                                                  num_layers=self.num_layers,
                                                  output_dim=self.hidden_dim,
                                                  bias=self.bias)
            channel_sum += self.hidden_dim

        self.convs = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=channel_sum, out_channels=self.output_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=1),
            nn.BatchNorm2d(self.output_dim)
        )  # input [B*S*E, H, W, (4)C] --> output [B*S*E, H, W, self.output_dim]


    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape [B, S, C, H, W]
        Returns
        -------

        """
        B, S, C, H, W = input_tensor.size()

        if self.order == 1:  # inverse the tensor along the elevation dimension
            input_tensor = input_tensor[:, ::-1, :, :, :]

        y_out = []
        h_state_fwd = None
        c_state_fwd = None
        for seq_idx in range(0, S):
            y_fwd, h_state_fwd, c_state_fwd = self.LSTM_layer(input_tensor[:, seq_idx, :, :, :],
                                                              h_state_fwd, c_state_fwd)
            # [B, C, H, W]
            y_out.append(y_fwd)
        y_out = torch.stack(y_out, dim=1)  # [B, S, C, H, W]

        if self.order == 2:
            y_inv_out = []
            h_state_inv = None
            c_state_inv = None
            for seq_idx in reversed(range(0, S)):
                y_inv, h_state_inv, c_state_inv = self.LSTM_layer_Inv(input_tensor[:, seq_idx, :, :, :],
                                                                      h_state_inv, c_state_inv)
                y_inv_out.append(y_inv)

            y_inv_out = torch.stack(y_inv_out, dim=1)  #  [B, S, C, H, W]
            y_out = torch.cat([y_out, y_inv_out], dim=2)

        y_out = y_out.view(B * S, -1, H, W)
        y_out = self.convs(y_out)
        # y_out = y_out.view(B, S, -1, H, W)

        return y_out


class VE_conv(nn.Module):
    ''' LSTM 2D for visibility and elevation estimation'''
    def __init__(self, seq_input_dim, hidden_dim, kernel_size, num_layers, ele_output_dim, seq_output_dim,
                 bias=True, seq_order=0, seq_fuse='LSTM', height_planes=8):
        # seq_fuse: conv or LSTM
        super(VE_conv, self).__init__()

        self.seq_input_dim = seq_input_dim  # int
        self.seq_output_dim = seq_output_dim  # int

        self.ele_input_dim = seq_output_dim
        self.ele_output_dim = ele_output_dim

        self.hidden_dim = hidden_dim  # int

        self.kernel_size = kernel_size  # tuple
        self.num_layers = num_layers  # int
        self.bias = bias  # bool

        if seq_fuse == 'conv':
            self.SeqNet = nn.Sequential(
                Reshape(),
                nn.Conv2d(in_channels=self.seq_input_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=self.seq_output_dim, kernel_size=(3, 3), stride=(1, 1), padding=1)
            )   # input [B, S, C, H, W] output [B*S, C, H, W]
        elif seq_fuse == 'LSTM':
            self.SeqNet = S_LSTM2D(self.seq_input_dim, hidden_dim, kernel_size, num_layers, self.seq_output_dim,
                 bias=True, order=seq_order)

        self.EleNet = nn.Sequential(
                nn.Conv2d(in_channels=self.ele_input_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=self.ele_output_dim, kernel_size=(3, 3), stride=(1, 1), padding=1)
            )   # input [B*E*S, C, H, W] output [B*E*S, C, H, W]

        self.conv_height = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=(self.ele_output_dim - 1)*height_planes, out_channels=height_planes, kernel_size=(3, 3), stride=(1, 1),
                      padding=1)
        )  # input [B, C, H, W] output [B, 1, H, W]

        # self.conv_height[-1].weight = nn.Parameter(torch.zeros(height_planes, (self.ele_output_dim - 1) * height_planes, 3, 3))
        # self.conv_height[-1].bias = nn.Parameter(torch.cat([torch.zeros(height_planes - 1), torch.ones(1)], dim=0))

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor:
            6-D Tensor of shape [B, S, E, C, H, W]
        Returns
        -------

        """
        B, S, E, C, H, W = input_tensor.size()

        # ======== First encode relationship between sequences ================
        seq_out = self.SeqNet(input_tensor.transpose(1, 2).contiguous().view(B*E, S, C, H, W))  # [B*E*S, C, H, W]
        ele_out = self.EleNet(seq_out) # [B*E*S, C, H, W]
        ele_out = ele_out.view(B, E, S, -1, H, W)
        vis, vis_feat = torch.split(ele_out, [1, self.ele_output_dim-1], dim=3)
        vis = nn.Softmax(dim=2)(vis).transpose(1, 2)  # [B, S, E, 1, H, W]
        vis_feat = torch.mean(vis_feat, dim=2)  # [B, E, C, H, W]

        vis_feat = vis_feat.view(B, -1, H, W)
        ele_feat = self.conv_height(vis_feat)  # [B, E, H, W]
        ele_prob = nn.Softmax(dim=1)(ele_feat.view(B, E, 1, H, W))

        return vis, ele_prob  # [B, S, E, 1, H, W], [B, E, 1, H, W]


class VisibilityFusion(nn.Module):
    ''' LSTM 2D for visibility and elevation estimation'''
    def __init__(self, seq_input_dim, hidden_dim, kernel_size, num_layers, seq_output_dim,
                 bias=True, seq_order=0, seq_fuse='LSTM'):
        # seq_fuse: conv or LSTM
        super(VisibilityFusion, self).__init__()

        self.seq_input_dim = seq_input_dim  # int
        self.seq_output_dim = seq_output_dim  # int

        self.hidden_dim = hidden_dim  # int

        self.kernel_size = kernel_size  # tuple
        self.num_layers = num_layers  # int
        self.bias = bias  # bool

        self.seq_fuse = seq_fuse
        

        if seq_fuse == 'Conv2D':
            self.SeqNet = nn.Sequential(
                Reshape(),
                nn.Conv2d(in_channels=self.seq_input_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=self.seq_output_dim, kernel_size=(3, 3), stride=(1, 1), padding=1)
            )   # input [B, S, C, H, W] output [B*S, C, H, W]
        elif seq_fuse == 'LSTM':
            self.SeqNet = S_LSTM2D(self.seq_input_dim, hidden_dim, kernel_size, num_layers, self.seq_output_dim,
                 bias=True, order=seq_order)
        elif seq_fuse == 'Conv3D':
            self.SeqNet = nn.Sequential(
                nn.Conv3d(in_channels=self.seq_input_dim, out_channels=hidden_dim,
                          kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.ReLU(),
                nn.Conv3d(in_channels=hidden_dim, out_channels=self.seq_output_dim,
                          kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            ) # [B, C, S, H, W] --> [B, C, S, H, W]

        self.convs = nn.Sequential(
                nn.Conv2d(in_channels=seq_output_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=1)
            )   # input [B*S, C, H, W] output [B*S, C, H, W]


    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor:
            6-D Tensor of shape [B, S, C, H, W]
        Returns
        -------

        """
        B, S, C, H, W = input_tensor.size()

        # ======== First encode relationship between sequences ================
        if self.seq_fuse == 'Conv3D':
            x = input_tensor.permute(0, 2, 1, 3, 4)
            y = self.SeqNet(x)  # [B, C, S, H, W]
            seq_out = y.permute(0, 2, 1, 3, 4).reshape(B*S, -1, H, W)
        else:
            seq_out = self.SeqNet(input_tensor)  # [B*S, C, H, W]
        vis = self.convs(seq_out).view(B, S, 1, H, W)

        vis = nn.Softmax(dim=1)(vis)  # [B, S, 1, H, W]

        fusion = torch.sum(input_tensor * vis, dim=1)  # [B, C, H, W]

        return fusion  # [B, C, H, W]


class Conv2DFusion(nn.Module):
    def __init__(self, seq_num, seq_input_dim, hidden_dim, kernel_size, num_layers, seq_output_dim,
                 bias=True, seq_order=0):
        # seq_fuse: conv or LSTM
        super(Conv2DFusion, self).__init__()

        self.seq_num = seq_num

        self.seq_input_dim = seq_input_dim  # int
        self.seq_output_dim = seq_output_dim  # int

        self.hidden_dim = hidden_dim  # int

        self.kernel_size = kernel_size  # tuple
        self.num_layers = num_layers  # int
        self.bias = bias  # bool

        self.convs0 = nn.Sequential(
            nn.Conv2d(in_channels=self.seq_input_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=seq_output_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=1)
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=seq_output_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=self.seq_input_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=1)
        )

    def forward(self, input_tensor):
        """
                Parameters
                ----------
                input_tensor:
                    6-D Tensor of shape [B, S, C, H, W]
                Returns
                -------

                """
        B, S, C, H, W = input_tensor.size()

        seq_input = input_tensor.reshape(B*S, C, H, W)

        # ======== First encode relationship between sequences ================
        seq_out = self.convs0(seq_input).reshape(B, S, -1, H, W)
        seq_out = torch.mean(seq_out, dim=1)
        fusion = self.convs(seq_out)  # [B, C, H, W]

        return fusion  # [B, C, H, W]

class Conv3DFusion(nn.Module):
    def __init__(self, seq_num, seq_input_dim, hidden_dim, kernel_size, num_layers, seq_output_dim,
                 bias=True, seq_order=0):
        # seq_fuse: conv or LSTM
        super(Conv3DFusion, self).__init__()

        self.seq_num = seq_num

        self.seq_input_dim = seq_input_dim  # int
        self.seq_output_dim = seq_output_dim  # int

        self.hidden_dim = hidden_dim  # int

        self.kernel_size = kernel_size  # tuple
        self.num_layers = num_layers  # int
        self.bias = bias  # bool

        self.Conv3D = nn.Sequential(
            nn.Conv3d(in_channels=seq_input_dim, out_channels=hidden_dim, kernel_size=(seq_num // 4, 3, 3),
                      stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(seq_num // 2, 3, 3),
                      stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=hidden_dim, out_channels=seq_output_dim, kernel_size=(2, 3, 3),
                      stride=(1, 1, 1), padding=(0, 1, 1)),
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=seq_output_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=self.seq_input_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=1)
        )

    def forward(self, input_tensor):
        """
                Parameters
                ----------
                input_tensor:
                    6-D Tensor of shape [B, S, C, H, W]
                Returns
                -------

                """
        B, S, C, H, W = input_tensor.size()

        # ======== First encode relationship between sequences ================
        seq_out = self.Conv3D(input_tensor.transpose(1, 2))  # [B, C, 1, H, W]
        fusion = self.convs(seq_out[:, :, 0, :, :])  # [B, C, H, W]

        return fusion  # [B, C, H, W]

class LSTMFusion(nn.Module):
    def __init__(self, seq_num, seq_input_dim, hidden_dim, kernel_size, num_layers, seq_output_dim,
                 bias=True, seq_order=0):
        # seq_fuse: conv or LSTM
        super(LSTMFusion, self).__init__()
        self.seq_num = seq_num

        self.seq_input_dim = seq_input_dim  # int
        self.seq_output_dim = seq_output_dim  # int

        self.hidden_dim = hidden_dim  # int

        self.kernel_size = kernel_size  # tuple
        self.num_layers = num_layers  # int
        self.bias = bias  # bool

        # if seq_fuse == 'Conv2D':
        #     self.SeqNet = nn.Sequential(
        #         Reshape_SC(),
        #         nn.Conv2d(in_channels=self.seq_input_dim * self.seq_num, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #         nn.ReLU(),
        #         nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #         nn.ReLU(),
        #         nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1),
        #                   padding=1),
        #         nn.ReLU(),
        #         nn.Conv2d(in_channels=hidden_dim, out_channels=self.seq_input_dim, kernel_size=(3, 3), stride=(1, 1), padding=1)
        #     )   # input [B, S, C, H, W] output [B, C, H, W]
        # elif seq_fuse == 'LSTM':
        self.SeqNet = S_LSTM2D(self.seq_input_dim, hidden_dim, kernel_size, num_layers, self.seq_output_dim,
             bias=True, order=seq_order)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=seq_output_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=self.seq_input_dim, kernel_size=(3, 3), stride=(1, 1), padding=1)
        )
        # elif seq_fuse == 'Conv3D':
        #     self.SeqNet = nn.Sequential(
        #
        #     )


    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor:
            6-D Tensor of shape [B, S, C, H, W]
        Returns
        -------

        """
        B, S, C, H, W = input_tensor.size()

        # ======== First encode relationship between sequences ================
        seq_out = self.SeqNet(input_tensor)  # [B*S, C, H, W]
        fusion = self.convs(seq_out.view(B, S, -1, H, W)[:, -1, :, :, :]) # [B, C, H, W]

        return fusion  # [B, C, H, W]