import torch
import torch_geometric
import torch_geometric.nn as gnn
import transformers as T
import util
from pojo import CMDConfig
import dataset_pojo
import sklearn.neighbors as skn
import mamba_ssm.models.mixer_seq_simple as mmm
import my_graph
import sklearn.ensemble as ske
import random
import math
import logging
import torch.nn.functional as F
import torch.nn as nn
import sklearn.metrics.pairwise as skmp
import torch_geometric.data as tgd


class BertProcess(torch.nn.Module):
    def __init__(self, cmd_config: CMDConfig):
        super(BertProcess, self).__init__()
        self.bert_model = T.AutoModel.from_pretrained(cmd_config.bert_path, output_hidden_states=True)
        self.bert_config = T.AutoConfig.from_pretrained(cmd_config.bert_path)
        self.layer_weights = torch.nn.Parameter(torch.ones(self.bert_config.n_layers) + 1)
        self.cmd_config = cmd_config
        self.fc = nn.Linear(cmd_config.mamba_hidden_state_f * (self.bert_config.n_layers + 1),
                            cmd_config.mamba_hidden_state_f)
        # 如果all 就不冻结了
        if "all" not in cmd_config.bert_unfreeze_layer:
            self.freeze_all()
            self.unfreeze_layers(cmd_config.bert_unfreeze_layer)

    def freeze_all(self):
        for name, param in self.bert_model.named_parameters():
            param.requires_grad = False

    def unfreeze_layers(self, layers):
        for layer in layers:
            for name, param in self.bert_model.named_parameters():
                if f"{layer}" in name:
                    param.requires_grad = True

    def forward(self, tokenizer, attention_mask, pool=None):
        bert_out = self.bert_model(tokenizer, attention_mask)
        if pool == "pool":
            return bert_out.last_hidden_state[:, 0, :]
        if pool == "all":
            return bert_out.hidden_states
        if pool == "all-max":
            hidden_states = bert_out.hidden_states[1:]
            max_pooled_output = torch.max(torch.stack(hidden_states), dim=0).values
            return max_pooled_output
        if pool == "all-add":
            hidden_states = bert_out.hidden_states[1:]
            add_pooled_output = hidden_states[0]
            for i in range(1, len(hidden_states), 1):
                add_pooled_output = hidden_states[i] + add_pooled_output
            return add_pooled_output
        if pool == "all-mean":
            hidden_states = bert_out.hidden_states[1:]
            averaged_hidden_states = torch.mean(torch.stack(hidden_states), dim=0)
            return averaged_hidden_states
        if pool == "all-weight":
            hidden_states = bert_out.hidden_states
            weighted_sum = sum(w * hidden_states[i] for i, w in zip(range(len(hidden_states)), self.layer_weights))
            return weighted_sum
        if pool == "fc":
            hidden_states = bert_out.hidden_states
            hidden_states = torch.cat(hidden_states, dim=-1)
            return self.fc(hidden_states)
        if pool == "emb":
            hidden_states = bert_out.hidden_states
            return hidden_states[0]
        return bert_out.last_hidden_state

    def reduce_x_sen(self, x_sen):
        return [item for sublist in x_sen for item in sublist]

    def x_sen_to_mask(self, x_sen):
        x_sen = self.reduce_x_sen(x_sen)
        sen_max_len = self.cmd_config.sen_max_length
        # Initialize mask with zeros
        mask = torch.zeros((len(x_sen), sen_max_len)).to(self.cmd_config.device)
        # Set the attention points to 1
        for i, sublist in enumerate(x_sen):
            for index in sublist:
                for index_item in index:
                    if index_item < sen_max_len:
                        mask[i, index_item] = 1
        # print(x_sen)
        # print(mask)
        return mask

    def fetch_word_feature(self, hidden_states, x_sen, seq=False):
        x_sen = self.reduce_x_sen(x_sen)
        hidden_states2 = []
        for batch_item_index, sentence_hidden_state in enumerate(hidden_states):
            # [[23], [56]]
            x_sen_item = x_sen[batch_item_index]
            x_sen_item1 = torch.LongTensor(x_sen_item[0]).to(hidden_states.device)
            x_sen_item2 = torch.LongTensor(x_sen_item[1]).to(hidden_states.device)
            hidden_state_1 = sentence_hidden_state[x_sen_item1].mean(0)
            hidden_state_2 = sentence_hidden_state[x_sen_item2].mean(0)
            if seq:
                hidden_state = torch.stack([hidden_state_1, hidden_state_2])
            else:
                hidden_state = torch.cat([hidden_state_1, hidden_state_2], dim=0)
            hidden_states2.append(hidden_state)
        hidden_states2 = torch.stack(hidden_states2)
        return hidden_states2


class OutProcess(torch.nn.Module):
    def __init__(self, channels):
        super(OutProcess, self).__init__()
        self.process = torch.nn.Sequential()
        for channel_i in range(len(channels) - 1):
            self.process.append(nn.Linear(channels[channel_i], channels[channel_i + 1]))
            if channel_i < len(channels) - 2:
                self.process.append(nn.LeakyReLU())
                self.process.append(nn.Dropout())

    def forward(self, hidden_states):
        return self.process(hidden_states)


class BertBaseSelector(torch.nn.Module):
    def __init__(self, cmd_config: CMDConfig):
        super(BertBaseSelector, self).__init__()
        self.model_index = -1
        self.cmd_config = cmd_config
        self.bert_encoder = BertProcess(cmd_config)
        self.out_process = OutProcess([768 * 2, 2])

    def forward(self, batch):
        _, reversed_data = batch
        x, y, edge_index, bert_x, bert_x_mask, sen, sen_mask, x_sen = reversed_data.x, reversed_data.y, reversed_data.edge_index, reversed_data.bert_x, reversed_data.bert_x_mask, reversed_data.sen, reversed_data.sen_mask, reversed_data.x_sen
        if self.training:
            sen, sen_mask, x_sen, y = util.neg_sample_0(self.cmd_config.neg_tag[self.model_index], sen, sen_mask, x_sen,
                                                        y,
                                                        self.bert_encoder.reduce_x_sen)
        sentence_hidden_states = self.bert_encoder(sen, sen_mask)
        hidden_states = self.bert_encoder.fetch_word_feature(sentence_hidden_states, x_sen)
        out = self.out_process(hidden_states)

        return out, y


class MambaBaseSelector(torch.nn.Module):
    def __init__(self, cmd_config: CMDConfig):
        super(MambaBaseSelector, self).__init__()
        self.model_index = -1
        self.cmd_config = cmd_config
        self.bert_encoder = BertProcess(cmd_config)
        self.mamba = mmm.Mamba(d_model=768)
        self.tcn = TemporalConvNet(num_inputs=768, num_channels=[1024])
        self.out_process = OutProcess([512, 2])
        self.graph_encoder1 = gnn.GraphConv(1024, 2048)
        self.graph_encoder2 = gnn.GraphConv(2048, 1024)
        self.graph_encoder3 = gnn.GraphConv(1024, 512)

    def forward(self, batch):
        _, reversed_data = batch
        x, y, edge_index, bert_x, bert_x_mask, sen, sen_mask, x_sen, graph_idx = reversed_data.x, reversed_data.y, reversed_data.edge_index, reversed_data.bert_x, reversed_data.bert_x_mask, reversed_data.sen, reversed_data.sen_mask, reversed_data.x_sen, reversed_data.batch
        if self.training:
            sen, sen_mask, x_sen, y, graph_idx = util.neg_sample_3(self.cmd_config.neg_tag[self.model_index], sen,
                                                                   sen_mask, x_sen,
                                                                   y, graph_idx,
                                                                   self.bert_encoder.reduce_x_sen)
        with torch.no_grad():
            bert_output_last_hidden_states = self.bert_encoder(sen, sen_mask)
        hidden_states = self.mamba(bert_output_last_hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.tcn(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.bert_encoder.fetch_word_feature(hidden_states, x_sen, True)
        phrase1_states = hidden_states[:, 0, :]
        phrase2_states = hidden_states[:, 1, :]
        phrase_states = torch.cat([phrase1_states, phrase2_states], dim=0)
        # 获取对应的图索引
        phrase_graph_idx = torch.cat([graph_idx, graph_idx], dim=0)

        k = self.cmd_config.cos_k
        graph_arr = []
        for bi in range(hidden_states.size(0)):
            dis1, idx1, c_graph_f1 = util.topk(phrase_states, phrase_graph_idx, bi, k)
            dis2, idx2, c_graph_f2 = util.topk(phrase_states, phrase_graph_idx, bi, k)

            # 第一个是自己
            idx1, idx2, dis1, dis2 = idx1[1:], idx2[1:], dis1[1:], dis2[1:]

            res1 = torch.index_select(c_graph_f1, 0, idx1)
            res2 = torch.index_select(c_graph_f2, 0, idx2)

            gen_x = [phrase1_states[bi], phrase2_states[bi]]
            gen_edge_index = [[0, 1], [1, 0]]
            gen_edge_w = [1, 1]
            gen_x_index = 2
            for node_i in range(res1.size(0)):
                gen_x.append(res1[node_i])
                gen_edge_index[0].append(0)
                gen_edge_index[1].append(gen_x_index)
                gen_edge_w.append(dis1[node_i])
                gen_x_index += 1
            for node_i in range(res2.size(0)):
                gen_x.append(res2[node_i])
                gen_edge_index[0].append(1)
                gen_edge_index[1].append(gen_x_index)
                gen_edge_w.append(dis2[node_i])
                gen_x_index += 1
            gen_x = torch.stack(gen_x)
            gen_edge_index = torch.tensor(gen_edge_index, dtype=torch.long, device=self.cmd_config.device)
            gen_edge_w = torch.tensor(gen_edge_w, dtype=torch.float, device=self.cmd_config.device)
            gen_graph = tgd.Data(x=gen_x, edge_index=gen_edge_index, edge_weight=gen_edge_w)
            graph_arr.append(gen_graph)
        graph_batch = tgd.Batch.from_data_list(graph_arr)
        # 没11个edge就是我们所需的edge
        hidden_states = self.graph_encoder1.forward(graph_batch.x, graph_batch.edge_index, graph_batch.edge_weight)
        hidden_states = self.graph_encoder2.forward(hidden_states, graph_batch.edge_index, graph_batch.edge_weight)
        hidden_states = self.graph_encoder3.forward(hidden_states, graph_batch.edge_index, graph_batch.edge_weight)

        hidden_states = gnn.global_mean_pool(hidden_states, graph_batch.batch)

        out = self.out_process(hidden_states)

        return out, y


class KNN():
    def __init__(self, cmd_config: CMDConfig):
        self.model_index = -1
        self.training = False
        self.cmd_config = cmd_config
        self.knn = skn.KNeighborsClassifier(n_neighbors=cmd_config.knn_n, weights='distance', p=cmd_config.knn_p)
        self.bert_encoder = BertProcess(cmd_config).to(self.cmd_config.device)

    def train(self):
        self.training = True
        self.bert_encoder.eval()

    def eval(self):
        self.training = False
        self.bert_encoder.eval()

    def __str__(self):
        return f"knn:{self.knn}"

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        if self.training:
            sen, sen_mask, x_sen, y = data.get_merge_x_y()
            if self.training:
                sen, sen_mask, x_sen, y = util.neg_sample_0(self.cmd_config.neg_tag[self.model_index], sen, sen_mask,
                                                            x_sen,
                                                            y,
                                                            self.bert_encoder.reduce_x_sen)
            with torch.no_grad():
                sen = sen.to(self.cmd_config.device)
                sen_mask = sen_mask.to(self.cmd_config.device)
                batch_size = 100
                feature = torch.tensor([], dtype=torch.float)
                for step in range(0, sen.size(0), batch_size):
                    temp = self.bert_encoder(sen[step:step + batch_size],
                                             sen_mask[step:step + batch_size]).cpu()
                    feature = torch.cat([feature, temp], dim=0)
                feature = self.bert_encoder.fetch_word_feature(feature, x_sen).cpu()
                x = feature

            self.knn.fit(x, y)
            return x, y
        else:
            _, reversed_data = data
            x, y, edge_index, bert_x, bert_x_mask, sen, sen_mask, x_sen = reversed_data.x, reversed_data.y, reversed_data.edge_index, reversed_data.bert_x, reversed_data.bert_x_mask, reversed_data.sen, reversed_data.sen_mask, reversed_data.x_sen
            with torch.no_grad():
                feature = self.bert_encoder(sen.to(self.cmd_config.device),
                                            sen_mask.to(self.cmd_config.device)).cpu()
                feature = self.bert_encoder.fetch_word_feature(feature, x_sen)

            prob_knn = torch.tensor(self.knn.predict_proba(feature), dtype=torch.float, device=self.cmd_config.device)
            prob = prob_knn
            return prob, y.to(self.cmd_config.device)


class GraphEncoder(torch.nn.Module):
    def __init__(self, cmd_config: CMDConfig):
        super(GraphEncoder, self).__init__()
        hidden_dim = cmd_config.mamba_hidden_state_f * 2
        self.graph1_1 = gnn.GraphConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            aggr="mean",
            bias=False)
        self.graph1_2 = gnn.GraphConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            aggr="mean",
            bias=False)

    def forward(self, x, edge_index):
        hidden_states = self.graph1_1(x, edge_index)

        hidden_states = self.graph1_2(hidden_states, edge_index)

        return hidden_states


class GraphNNSelector(torch.nn.Module):
    def __init__(self, cmd_config: CMDConfig):
        super(GraphNNSelector, self).__init__()
        self.cmd_config = cmd_config
        self.model_index = -1
        self.bert_encoder = BertProcess(cmd_config)
        self.hidden_layers = GraphEncoder(cmd_config)
        self.out_process = OutProcess([768 * 2, 2])

    def forward(self, batch):
        _, reversed_data = batch
        x, y, edge_index, bert_x, bert_x_mask, sen, sen_mask, x_sen = reversed_data.x, reversed_data.y, reversed_data.edge_index, reversed_data.bert_x, reversed_data.bert_x_mask, reversed_data.sen, reversed_data.sen_mask, reversed_data.x_sen
        if self.training:
            sen, sen_mask, x_sen, y, edge_index = util.neg_sample_4(self.cmd_config.neg_tag[self.model_index], sen,
                                                                   sen_mask, x_sen,
                                                                   y, edge_index,
                                                                   self.bert_encoder.reduce_x_sen)
        with torch.no_grad():
            hidden_states = self.bert_encoder(sen, sen_mask)
        hidden_states = self.bert_encoder.fetch_word_feature(hidden_states, x_sen)
        if hidden_states.size(0) == 1:
            logging.warning(f"{reversed_data.file_name}出现了问题，可能会导致BatchNorm出现问题")
        hidden_states = self.hidden_layers(hidden_states, edge_index)
        out = self.out_process(hidden_states)

        return out, y


class NormGNN(nn.Module):
    def __init__(self, cmd_config: CMDConfig):
        super(NormGNN, self).__init__()
        self.cmd_config = cmd_config
        self.model_index = -1
        self.bert_encoder = BertProcess(cmd_config)
        self.graph1 = gnn.GraphConv(768, 768 * 2)
        self.graph2 = gnn.GraphConv(768 * 2, 768)
        self.out = OutProcess([768 * 2, 2])

    def forward(self, batch):
        data, reversed_data = batch
        x, x_mask, edge_index, edge_type = data.bert_x, data.bert_x_mask, data.edge_index, data.edge_type
        with torch.no_grad():
            bert_out = self.bert_encoder(x, x_mask, "pool")
        hidden_states = self.graph1(bert_out, edge_index)
        hidden_states = self.graph2(hidden_states, edge_index)
        hidden_states = torch.cat([hidden_states[edge_index[0,:]],hidden_states[edge_index[1,:]]],dim=-1)
        out = self.out(hidden_states)
        return out,edge_type



# 定义 Chomp1d 模块，用于去除多余的填充
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# 定义单个时间块（Temporal Block）
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 将卷积层、Chomp1d、ReLU和Dropout连接成一个网络
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 如果输入和输出通道数不同，则使用1x1卷积进行降采样
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# 定义整个时间卷积网络（TCN）
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 扩张系数随层数指数增长
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CrossEntropyLossWithL1(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', l1_lambda=0.01):
        super(CrossEntropyLossWithL1, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight, size_average, ignore_index,
                                           reduce, reduction)
        self.l1_lambda = l1_lambda

    def forward(self, input, target):
        ce_loss = self.ce_loss(input, target)
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))

        loss = ce_loss + self.l1_lambda * l1_loss
        return loss
