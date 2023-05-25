import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset


class DataLoader(Dataset):
    def __init__(self, codes, sensitive, target, m):
        self.dataset = codes
        self.sensitive = sensitive
        self.target = target
        self.m = m

    def __getitem__(self, index):

        code = self.dataset[index]
        sensitive = int(self.sensitive[index])
        target = int(self.target[index])
        m = int(self.m[index])
        return torch.Tensor(code), torch.LongTensor([sensitive]), torch.Tensor([target]), m

    def __len__(self):
        return len(self.dataset)


class LocationAttention(nn.Module):

    def __init__(self, hidden_size, device):
        super(LocationAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_value_ori_func = nn.Linear(self.hidden_size, 1)
        self.device = device

    def forward(self, input_data):
        # shape of input_data: <n_batch, n_seq, hidden_size>
        n_batch, n_seq, hidden_size = input_data.shape
        # shape of reshape_feat: <n_batch*n_seq, hidden_size>
        reshape_feat = input_data.reshape(n_batch * n_seq, hidden_size)
        # shape of attention_value_ori: <n_batch*n_seq, 1>
        attention_value_ori = torch.exp(self.attention_value_ori_func(reshape_feat))
        # shape of attention_value_format: <n_batch, 1, n_seq>
        attention_value_format = attention_value_ori.reshape(n_batch, n_seq).unsqueeze(1)
        # shape of ensemble flag format: <1, n_seq, n_seq>
        # if n_seq = 3, ensemble_flag_format can get below flag data
        #  [[[ 0  0  0
        #      1  0  0
        #      1  1  0 ]]]
        ensemble_flag_format = torch.triu(torch.ones([n_seq, n_seq]), diagonal=1).permute(1, 0).unsqueeze(0).to(
            self.device)
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-9
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * ensemble_flag_format
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        attention_weight_format = each_attention_value / accumulate_attention_value
        # shape of _extend_attention_weight_format: <n_batch, n_seq, n_seq, 1>
        _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _weighted_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _weighted_input_data = _extend_attention_weight_format * _extend_input_data
        # shape of weighted_output: <n_batch, n_seq, hidden_size>
        weighted_output = torch.sum(_weighted_input_data, 2)
        return weighted_output


class GeneralAttention(nn.Module):

    def __init__(self, hidden_size, device):
        super(GeneralAttention, self).__init__()
        self.hidden_size = hidden_size
        self.correlated_value_ori_func = nn.Linear(self.hidden_size, self.hidden_size)
        self.device = device

    def forward(self, input_data):
        # shape of input_data: <n_batch, n_seq, hidden_size>
        n_batch, n_seq, hidden_size = input_data.shape
        # shape of reshape_feat: <n_batch*n_seq, hidden_size>
        reshape_feat = input_data.reshape(n_batch * n_seq, hidden_size)
        # shape of correlated_value_ori: <n_batch, n_seq, hidden_size>
        correlated_value_ori = self.correlated_value_ori_func(reshape_feat).reshape(n_batch, n_seq, hidden_size)
        # shape of _extend_correlated_value_ori: <n_batch, n_seq, 1, hidden_size>
        _extend_correlated_value_ori = correlated_value_ori.unsqueeze(-2)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _extend_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _correlat_value = _extend_correlated_value_ori * _extend_input_data
        # shape of attention_value_format: <n_batch, n_seq, n_seq>
        attention_value_format = torch.exp(torch.sum(_correlat_value, dim=-1))
        # shape of ensemble flag format: <1, n_seq, n_seq>
        # if n_seq = 3, ensemble_flag_format can get below flag data
        #  [[[ 0  0  0
        #      1  0  0
        #      1  1  0 ]]]
        ensemble_flag_format = torch.triu(torch.ones([n_seq, n_seq]), diagonal=1).permute(1, 0).unsqueeze(0).to(
            self.device)
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-10
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * ensemble_flag_format
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        attention_weight_format = each_attention_value / accumulate_attention_value
        # shape of _extend_attention_weight_format: <n_batch, n_seq, n_seq, 1>
        _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _weighted_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _weighted_input_data = _extend_attention_weight_format * _extend_input_data
        # shape of weighted_output: <n_batch, n_seq, hidden_size>
        weighted_output = torch.sum(_weighted_input_data, 2)
        return weighted_output


class ConcatenationAttention(nn.Module):
    def __init__(self, hidden_size, attention_dim=16, device=None):
        super(ConcatenationAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.attention_map_func = nn.Linear(2 * self.hidden_size, self.attention_dim)
        self.activate_func = nn.Tanh()
        self.correlated_value_ori_func = nn.Linear(self.attention_dim, 1)
        self.device = device

    def forward(self, input_data):
        # shape of input_data: <n_batch, n_seq, hidden_size>
        n_batch, n_seq, hidden_size = input_data.shape
        # shape of _extend_input_data: <n_batch, n_seq, 1, hidden_size>
        _extend_input_data_f = input_data.unsqueeze(-2)
        # shape of _repeat_extend_correlated_value_ori: <n_batch, n_seq, n_seq, hidden_size>
        _repeat_extend_input_data_f = _extend_input_data_f.repeat(1, 1, n_seq, 1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data_b = input_data.unsqueeze(1)
        # shape of _repeat_extend_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _repeat_extend_input_data_b = _extend_input_data_b.repeat(1, n_seq, 1, 1)
        # shape of _concate_value: <n_batch, n_seq, n_seq, 2 * hidden_size>
        _concate_value = torch.cat([_repeat_extend_input_data_f, _repeat_extend_input_data_b], dim=-1)
        # shape of _correlat_value: <n_batch, n_seq, n_seq>
        _correlat_value = self.activate_func(self.attention_map_func(_concate_value.reshape(-1, 2 * hidden_size)))
        _correlat_value = self.correlated_value_ori_func(_correlat_value).reshape(n_batch, n_seq, n_seq)
        # shape of attention_value_format: <n_batch, n_seq, n_seq>
        attention_value_format = torch.exp(_correlat_value)
        # shape of ensemble flag format: <1, n_seq, n_seq>
        # if n_seq = 3, ensemble_flag_format can get below flag data
        #  [[[ 0  0  0
        #      1  0  0
        #      1  1  0 ]]]
        ensemble_flag_format = torch.triu(torch.ones([n_seq, n_seq]), diagonal=1).permute(1, 0).unsqueeze(0).to(
            self.device)
        # shape of accumulate_attention_value: <n_batch, n_seq, 1>
        accumulate_attention_value = torch.sum(attention_value_format * ensemble_flag_format, -1).unsqueeze(-1) + 1e-10
        # shape of each_attention_value: <n_batch, n_seq, n_seq>
        each_attention_value = attention_value_format * ensemble_flag_format
        # shape of attention_weight_format: <n_batch, n_seq, n_seq>
        attention_weight_format = each_attention_value / accumulate_attention_value
        # shape of _extend_attention_weight_format: <n_batch, n_seq, n_seq, 1>
        _extend_attention_weight_format = attention_weight_format.unsqueeze(-1)
        # shape of _extend_input_data: <n_batch, 1, n_seq, hidden_size>
        _extend_input_data = input_data.unsqueeze(1)
        # shape of _weighted_input_data: <n_batch, n_seq, n_seq, hidden_size>
        _weighted_input_data = _extend_attention_weight_format * _extend_input_data
        # shape of weighted_output: <n_batch, n_seq, hidden_size>
        weighted_output = torch.sum(_weighted_input_data, 2)
        return weighted_output


class Dipole(nn.Module):
    def __init__(self,
                 input_size=None,
                 embed_size=144,
                 hidden_size=144,
                 output_size=144,
                 bias=True,
                 dropout=0.3,
                 batch_first=True,
                 label_size=1,
                 attention_type='location_based',
                 attention_dim=4,
                 device=None):
        super(Dipole, self).__init__()
        assert input_size != None and isinstance(input_size, int), 'fill in correct input_size'

        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.label_size = label_size

        self.embed_func = nn.Linear(self.input_size, self.embed_size)
        self.rnn_model = nn.GRU(input_size=embed_size,
                                hidden_size=hidden_size,
                                bias=bias,
                                num_layers=2,
                                dropout=dropout,
                                bidirectional=True,
                                batch_first=batch_first)
        if attention_type == 'location_based':
            self.attention_func = LocationAttention(2 * hidden_size, device)
        elif attention_type == 'general':
            self.attention_func = GeneralAttention(2 * hidden_size, device)
        elif attention_type == 'concatenation_based':
            self.attention_func = ConcatenationAttention(2 * hidden_size, attention_dim=attention_dim, device=device)
        else:
            raise Exception('fill in correct attention_type, [location_based, general, concatenation_based]')
        self.output_func = nn.Linear(4 * hidden_size, self.output_size)
        self.output_activate = nn.Tanh()
        self.classifier = nn.Sequential(
            nn.Linear(self.output_size, 1)
        )
        self.sens_class = nn.Sequential(
            nn.Linear(self.output_size, self.output_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.output_size, self.output_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.output_size, self.output_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.output_size, 5)
        )

    def forward(self, input_data, M):

        """

        Parameters

        ----------
        input_data = {
                      'X': shape (batchsize, n_timestep, n_featdim)
                      'M': shape (batchsize, n_timestep)
                      'cur_M': shape (batchsize, n_timestep)
                      'T': shape (batchsize, n_timestep)
                     }

        Return

        ----------

        all_output, shape (batchsize, n_timestep, n_labels)

            predict output of each time step

        cur_output, shape (batchsize, n_labels)

            predict output of last time step

        """

        X = input_data
        batchsize, n_timestep, n_orifeatdim = X.shape
        _ori_X = X.view(-1, n_orifeatdim)
        _embed_X = self.embed_func(_ori_X)
        _embed_X = _embed_X.reshape(batchsize, n_timestep, self.embed_size)
        _embed_F, _ = self.rnn_model(_embed_X)
        _embed_F_w = self.attention_func(_embed_F)
        _mix_F = torch.cat([_embed_F, _embed_F_w], dim=-1)
        _mix_F_reshape = _mix_F.view(-1, 4 * self.hidden_size)
        outputs = self.output_activate(self.output_func(_mix_F_reshape)).reshape(batchsize, n_timestep,
                                                                                 self.output_size)
        n_batchsize, n_timestep, output_size = outputs.shape

        all_output = self.classifier(outputs[:, -1, :])
        sens = self.sens_class(outputs[:, -1, :])
        return outputs, all_output, sens
