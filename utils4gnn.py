import matplotlib.pyplot as plt
import seaborn as sns

import dgl
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import time
from dgl.nn.pytorch import GATConv, GATv2Conv, GINConv, SAGEConv, PNAConv, EdgeConv
from layers import WeightedGraphConv, MaxRelativeGraphConv
from networks import (
    Model_GAT_LSTM,
    Model_GATV2_LSTM,
    Model_GIN_LSTM,
    Model_SAGE_LSTM,
    Model_WConv_LSTM,
    Model_PNA_LSTM,
    Model_EdgeConv_LSTM,
    Model_MRConv_LSTM
)

IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'
LOW_FEAT_NUM = 514
HIGH_FEAT_NUM = 281
OUTPUT_FEAT_NUM = 128
# DISASTER_TYPE = {
#     'normal': 0,
#     'flood': 1,
#     'tsunami': 2,
#     'hurricane': 3,
#     'tornado': 4,
#     'fire': 5,
#     'wildfire': 6,
#     'bushfire': 7,
#     'volcano': 8,
#     'earthquake': 9,
#     'debrisflow': 10,
#     'landslide': 11,
#     'explosion': 12,
# }
DISASTER_TYPE = {
    'normal': 0,
    'anomaly': 1,
}

#GATV2
class SimpleGATV2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads, rep_length):
        super(SimpleGATV2, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        self.gat_layers.append(GATv2Conv(in_dim, hidden_dim, heads[0]))
        self.norm_layers.append(nn.LayerNorm(hidden_dim * heads[0],))

        for l in range(1, num_layers - 1):
            self.gat_layers.append(
                GATv2Conv(hidden_dim * heads[l-1], hidden_dim, heads[l]))
            self.norm_layers.append(nn.LayerNorm(hidden_dim * heads[l]))

        self.gat_layers.append(
            GATv2Conv(hidden_dim * heads[-2], out_dim, heads[-1]))
        self.norm_layers.append(nn.LayerNorm(out_dim * heads[-1]))


    def forward(self, g, features):
        h = features
        pred_list = []
        g_list = dgl.unbatch(g)
        update_g_list = []
        for iter_g in g_list:
            iter_g = dgl.add_self_loop(iter_g)
            h = iter_g.ndata['feat'] 
            for l in range(self.num_layers):
                h = self.gat_layers[l](iter_g, h).flatten(1)
                h = self.norm_layers[l](h)
                h = F.elu(h)
            hg = torch.mean(h, dim=0)
            pred_list.append(hg)
            iter_g.ndata['feat'] = h
            update_g_list.append(iter_g)

        update_g_batch = dgl.batch(update_g_list)
        return update_g_batch
    
#GAT
class SimpleGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads, rep_length):
        super(SimpleGAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_dim, hidden_dim, heads[0]))
        self.norm_layers.append(nn.LayerNorm(hidden_dim * heads[0],))
        for l in range(1, num_layers - 1):
            self.gat_layers.append(
                GATConv(hidden_dim * heads[l-1], hidden_dim, heads[l]))
            self.norm_layers.append(nn.LayerNorm(hidden_dim * heads[l]))
        self.gat_layers.append(
            GATConv(hidden_dim * heads[-2], out_dim, heads[-1]))
        self.norm_layers.append(nn.LayerNorm(out_dim * heads[-1]))


    def forward(self, g, features):
        h = features
        pred_list = []
        g_list = dgl.unbatch(g)
        update_g_list = []
        for iter_g in g_list:
            iter_g = dgl.add_self_loop(iter_g)
            h = iter_g.ndata['feat']
            for l in range(self.num_layers):
                h = self.gat_layers[l](iter_g, h).flatten(1)
                h = self.norm_layers[l](h)
                h = F.elu(h)
            hg = torch.mean(h, dim=0)
            pred_list.append(hg)
            iter_g.ndata['feat'] = h
            update_g_list.append(iter_g)

        update_g_batch = dgl.batch(update_g_list)
        return update_g_batch


class SimpleGIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super(SimpleGIN, self).__init__()
        self.num_layers = num_layers
        self.gin_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.gin_layers.append(GINConv(nn.Linear(in_dim, hidden_dim), 'max'))
        self.norm_layers.append(nn.LayerNorm(hidden_dim,))
        for l in range(1, num_layers - 1):
            self.gin_layers.append(
                GINConv(nn.Linear(hidden_dim, hidden_dim), 'max'))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
        self.gin_layers.append(
            GINConv(nn.Linear(hidden_dim, out_dim), 'max'))
        self.norm_layers.append(nn.LayerNorm(out_dim))

    def forward(self, g, features):
        h = features
        pred_list = []
        g_list = dgl.unbatch(g)
        update_g_list = []
        for iter_g in g_list:
            iter_g = dgl.add_self_loop(iter_g)
            h = iter_g.ndata['feat']  
            for l in range(self.num_layers):
                h = self.gin_layers[l](iter_g, h).flatten(1)
                h = self.norm_layers[l](h)
                h = F.elu(h)
            hg = torch.mean(h, dim=0)
            pred_list.append(hg)
            iter_g.ndata['feat'] = h
            update_g_list.append(iter_g)

        update_g_batch = dgl.batch(update_g_list)
        return update_g_batch
    

class SimpleSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super(SimpleSAGE, self).__init__()
        self.num_layers = num_layers
        self.sage_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.sage_layers.append(SAGEConv(in_dim, hidden_dim,aggregator_type='mean'))
        self.norm_layers.append(nn.LayerNorm(hidden_dim,))
        for l in range(1, num_layers - 1):
            self.sage_layers.append(
                SAGEConv(hidden_dim, hidden_dim,aggregator_type='mean'))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
        self.sage_layers.append(
            SAGEConv(hidden_dim, out_dim, aggregator_type='mean'))
        self.norm_layers.append(nn.LayerNorm(out_dim))


    def forward(self, g, features):
        h = features
        pred_list = []
        g_list = dgl.unbatch(g)
        update_g_list = []
        for iter_g in g_list:
            iter_g = dgl.add_self_loop(iter_g)
            h = iter_g.ndata['feat']  
            for l in range(self.num_layers):
                h = self.sage_layers[l](iter_g, h).flatten(1)
                h = self.norm_layers[l](h)
                h = F.elu(h)
            hg = torch.mean(h, dim=0)
            pred_list.append(hg)
            iter_g.ndata['feat'] = h
            update_g_list.append(iter_g)

        update_g_batch = dgl.batch(update_g_list)
        return update_g_batch

class SimpleWConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super(SimpleWConv, self).__init__()
        self.num_layers = num_layers
        self.wconv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.wconv_layers.append(WeightedGraphConv(in_dim, hidden_dim))
        self.norm_layers.append(nn.LayerNorm(hidden_dim,))
        for l in range(1, num_layers - 1):
            self.wconv_layers.append(
                WeightedGraphConv(hidden_dim, hidden_dim))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
        self.wconv_layers.append(
            WeightedGraphConv(hidden_dim, out_dim))
        self.norm_layers.append(nn.LayerNorm(out_dim))


    def forward(self, g, features):
        h = features
        pred_list = []
        g_list = dgl.unbatch(g)
        update_g_list = []
        for iter_g in g_list:
            iter_g = dgl.add_self_loop(iter_g)
            h = iter_g.ndata['feat']
            for l in range(self.num_layers):
                h = self.wconv_layers[l](iter_g, h)
                h = self.norm_layers[l](h)
                h = F.elu(h)
            hg = torch.mean(h, dim=0)
            pred_list.append(hg)
            iter_g.ndata['feat'] = h
            update_g_list.append(iter_g)

        update_g_batch = dgl.batch(update_g_list)
        return update_g_batch

class SimplePNA(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers,deg = 4, aggregators = ['max','min','std','var'], scalers = ['identity', 'amplification','attenuation']):
        super(SimplePNA, self).__init__()
        self.num_layers = num_layers
        self.pna_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.pna_layers.append(PNAConv(in_dim, hidden_dim, aggregators, scalers, delta=deg))
        self.norm_layers.append(nn.LayerNorm(hidden_dim,))
        for l in range(1, num_layers - 1):
            self.pna_layers.append(
                PNAConv(hidden_dim, hidden_dim, aggregators, scalers, delta=deg))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
        self.pna_layers.append(
            PNAConv(hidden_dim, out_dim, aggregators, scalers, delta=deg))
        self.norm_layers.append(nn.LayerNorm(out_dim))


    def forward(self, g, features):
        h = features
        pred_list = []
        g_list = dgl.unbatch(g)
        update_g_list = []
        for iter_g in g_list:
            iter_g = dgl.add_self_loop(iter_g)
            h = iter_g.ndata['feat']
            for l in range(self.num_layers):
                h = self.pna_layers[l](iter_g, h).flatten(1)
                h = self.norm_layers[l](h)
                h = F.elu(h)
            hg = torch.mean(h, dim=0)
            pred_list.append(hg)
            iter_g.ndata['feat'] = h
            update_g_list.append(iter_g)

        update_g_batch = dgl.batch(update_g_list)
        return update_g_batch
    

class SimpleEdgeConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super(SimpleEdgeConv, self).__init__()
        self.num_layers = num_layers
        self.gin_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.gin_layers.append(EdgeConv(in_dim, hidden_dim))
        self.norm_layers.append(nn.LayerNorm(hidden_dim,))
        for l in range(1, num_layers - 1):
            self.gin_layers.append(
                EdgeConv(hidden_dim, hidden_dim))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
        self.gin_layers.append(
            EdgeConv(hidden_dim, out_dim))
        self.norm_layers.append(nn.LayerNorm(out_dim))

    def forward(self, g, features):
        h = features
        pred_list = []
        g_list = dgl.unbatch(g)
        update_g_list = []
        for iter_g in g_list:
            iter_g = dgl.add_self_loop(iter_g)
            h = iter_g.ndata['feat']  
            for l in range(self.num_layers):
                h = self.gin_layers[l](iter_g, h).flatten(1)
                h = self.norm_layers[l](h)
                h = F.elu(h)
            hg = torch.mean(h, dim=0)
            pred_list.append(hg)
            iter_g.ndata['feat'] = h
            update_g_list.append(iter_g)

        update_g_batch = dgl.batch(update_g_list)
        return update_g_batch


class SimpleMRConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super(SimpleMRConv, self).__init__()
        self.num_layers = num_layers
        self.mrconv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        self.mrconv_layers.append(MaxRelativeGraphConv(in_dim, hidden_dim))
        self.norm_layers.append(nn.LayerNorm(hidden_dim,))

        for l in range(1, num_layers - 1):
            self.mrconv_layers.append(
                MaxRelativeGraphConv(hidden_dim, hidden_dim))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))

        self.mrconv_layers.append(
            MaxRelativeGraphConv(hidden_dim, out_dim))
        self.norm_layers.append(nn.LayerNorm(out_dim))

    def forward(self, g, features):
        h = features
        pred_list = []
        g_list = dgl.unbatch(g)
        update_g_list = []
        for iter_g in g_list:
            iter_g = dgl.add_self_loop(iter_g)
            h = iter_g.ndata['feat']
            for l in range(self.num_layers):
                h = self.mrconv_layers[l](iter_g, h)
                h = self.norm_layers[l](h)
                h = F.elu(h)
            hg = torch.mean(h, dim=0)
            pred_list.append(hg)
            iter_g.ndata['feat'] = h
            update_g_list.append(iter_g)

        update_g_batch = dgl.batch(update_g_list)
        return update_g_batch
    

class MultiScaleHGPSL(nn.Module):
    def __init__(self, num_layers, dropout_rate, pool_ratio, lamb, sample, \
                 layer_type = 'WConv', agg_mode = 'sum', ass=True, \
                 mod='high', fusion_mode='AFM'):
        super(MultiScaleHGPSL, self).__init__()


        self.num_layers = num_layers 
        self.dropout_rate = dropout_rate  
        self.pool_ratio = pool_ratio  
        self.lamb = lamb  
        self.sample = sample  
        self.layer_type = layer_type  
        self.agg_mode = agg_mode  
        self.ass = ass
        self.mod = mod
        self.fusion_mode = fusion_mode

        self.build_low_level_graph()  

        self.build_high_level_graph()  


        self.att_conv_low = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.att_conv_high = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )


        self.global_pool=nn.AdaptiveAvgPool2d(1)


        self.att_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

 
        self.fc2 = nn.Conv2d(in_channels=1,out_channels=2,kernel_size = 1,stride = 1,bias=False)  # 升维


        self.softmax = nn.Softmax(dim=0)


        self.MLP = nn.Sequential(
            nn.Linear(128, 2),  
            nn.Dropout(0.2),  
            nn.Sigmoid()  
        )

    def build_low_level_graph(self):
        if self.layer_type == 'WConv':
            self.pre_model = SimpleWConv(in_dim = LOW_FEAT_NUM, hidden_dim = 32, out_dim = 128, num_layers = 3).to(DEVICE)
            self.low_model = Model_WConv_LSTM(in_feat= 128, out_feat=128, hid_feat=32,
                    conv_layers=self.num_layers, dropout=self.dropout_rate, pool_ratio=self.pool_ratio,
                    lamb=self.lamb, sample='true', agg_mode=self.agg_mode).to(DEVICE)
            
        elif self.layer_type == 'GATv2':
            self.heads = [4] * 3
            self.pre_model = SimpleGATV2(in_dim = LOW_FEAT_NUM, hidden_dim = 32, out_dim = 32, num_layers = 3, heads=self.heads, rep_length = 256).to(DEVICE)
            self.low_model = Model_GATV2_LSTM(in_feat= OUTPUT_FEAT_NUM, out_feat=128, hid_feat=32,
                    conv_layers=self.num_layers, dropout=self.dropout_rate, pool_ratio=self.pool_ratio,
                    lamb=self.lamb, sample='true', num_heads = 4, agg_mode=self.agg_mode).to(DEVICE)
            
        elif self.layer_type == 'GAT':
            self.heads = [4] * 3
            self.pre_model = SimpleGAT(in_dim = LOW_FEAT_NUM, hidden_dim = 32, out_dim = 32, num_layers = 3, heads=self.heads, rep_length = 256).to(DEVICE)
            self.low_model = Model_GAT_LSTM(in_feat= OUTPUT_FEAT_NUM, out_feat=128, hid_feat=32,
                    conv_layers=self.num_layers, dropout=self.dropout_rate, pool_ratio=self.pool_ratio,
                    lamb=self.lamb, sample='true', num_heads = 4, agg_mode=self.agg_mode).to(DEVICE)
        
        elif self.layer_type == 'GIN':
            self.pre_model = SimpleGIN(in_dim = LOW_FEAT_NUM, hidden_dim = 32, out_dim = 128, num_layers = 3).to(DEVICE)
            self.low_model = Model_GIN_LSTM(in_feat= 128, out_feat=128, hid_feat=32,
                    conv_layers=self.num_layers, dropout=self.dropout_rate, pool_ratio=self.pool_ratio,
                    lamb=self.lamb, sample='true', agg_mode=self.agg_mode).to(DEVICE)
            
        elif self.layer_type == 'SAGE':
            self.pre_model = SimpleSAGE(in_dim = LOW_FEAT_NUM, hidden_dim = 32, out_dim = 128, num_layers = 3).to(DEVICE)
            self.low_model = Model_SAGE_LSTM(in_feat= 128, out_feat=128, hid_feat=32,
                    conv_layers=self.num_layers, dropout=self.dropout_rate, pool_ratio=self.pool_ratio,
                    lamb=self.lamb, sample='true', agg_mode=self.agg_mode).to(DEVICE)
        
        elif self.layer_type == 'PNA':
            self.pre_model = SimplePNA(in_dim = LOW_FEAT_NUM, hidden_dim = 32, out_dim = 128, num_layers = 3).to(DEVICE)
            self.low_model = Model_PNA_LSTM(in_feat= 128, out_feat=128, hid_feat=32,
                    conv_layers=self.num_layers, dropout=self.dropout_rate, pool_ratio=self.pool_ratio,
                    lamb=self.lamb, sample='true', agg_mode=self.agg_mode).to(DEVICE)
        
        elif self.layer_type == 'EConv':
            self.pre_model = SimpleEdgeConv(in_dim = LOW_FEAT_NUM, hidden_dim = 32, out_dim = 128, num_layers = 3).to(DEVICE)
            self.low_model = Model_EdgeConv_LSTM(in_feat= 128, out_feat=128, hid_feat=32,
                    conv_layers=self.num_layers, dropout=self.dropout_rate, pool_ratio=self.pool_ratio,
                    lamb=self.lamb, sample='true', agg_mode=self.agg_mode).to(DEVICE)
        
        elif self.layer_type == 'MRConv':
            self.pre_model = SimpleMRConv(in_dim = LOW_FEAT_NUM, hidden_dim = 32, out_dim = 128, num_layers = 3).to(DEVICE)
            self.low_model = Model_MRConv_LSTM(in_feat= 128, out_feat=128, hid_feat=32,
                    conv_layers=self.num_layers, dropout=self.dropout_rate, pool_ratio=self.pool_ratio,
                    lamb=self.lamb, sample='true', agg_mode=self.agg_mode).to(DEVICE)

    def build_high_level_graph(self):
        if self.layer_type == 'WConv':
            self.high_model = Model_WConv_LSTM(in_feat= HIGH_FEAT_NUM + OUTPUT_FEAT_NUM, out_feat=128, hid_feat=32,
                    conv_layers=self.num_layers, dropout=self.dropout_rate, pool_ratio=self.pool_ratio,
                    lamb=self.lamb, sample='true', num_heads = 4, agg_mode=self.agg_mode).to(DEVICE)
            
        elif self.layer_type == 'GATv2':
            self.high_model = Model_GATV2_LSTM(in_feat=HIGH_FEAT_NUM + OUTPUT_FEAT_NUM, out_feat=128, hid_feat=64,
                    conv_layers=self.num_layers, dropout=0, pool_ratio=0.5,
                    lamb=self.lamb, sample='true', num_heads = 4, agg_mode=self.agg_mode).to(DEVICE)
            
        elif self.layer_type == 'GAT':
            self.high_model = Model_GAT_LSTM(in_feat=HIGH_FEAT_NUM + OUTPUT_FEAT_NUM, out_feat=128, hid_feat=64,
                    conv_layers=self.num_layers, dropout=0, pool_ratio=0.5,
                    lamb=self.lamb, sample='true', num_heads = 4, agg_mode=self.agg_mode).to(DEVICE)
        
        elif self.layer_type == 'GIN':
            self.high_model = Model_GIN_LSTM(in_feat= HIGH_FEAT_NUM + OUTPUT_FEAT_NUM, out_feat=128, hid_feat=32,
                    conv_layers=self.num_layers, dropout=self.dropout_rate, pool_ratio=self.pool_ratio,
                    lamb=self.lamb, sample='true', num_heads = 4, agg_mode=self.agg_mode).to(DEVICE)
            
        elif self.layer_type == 'SAGE':
            self.high_model = Model_SAGE_LSTM(in_feat= HIGH_FEAT_NUM + OUTPUT_FEAT_NUM, out_feat=128, hid_feat=32,
                    conv_layers=self.num_layers, dropout=self.dropout_rate, pool_ratio=self.pool_ratio,
                    lamb=self.lamb, sample='true', agg_mode=self.agg_mode).to(DEVICE)
        
        elif self.layer_type == 'PNA':
            self.high_model = Model_PNA_LSTM(in_feat= HIGH_FEAT_NUM + OUTPUT_FEAT_NUM, out_feat=128, hid_feat=32,
                    conv_layers=self.num_layers, dropout=self.dropout_rate, pool_ratio=self.pool_ratio,
                    lamb=self.lamb, sample='true', agg_mode=self.agg_mode).to(DEVICE)
        
        elif self.layer_type == 'EConv':
            self.high_model = Model_EdgeConv_LSTM(in_feat= HIGH_FEAT_NUM + OUTPUT_FEAT_NUM, out_feat=128, hid_feat=32,
                    conv_layers=self.num_layers, dropout=self.dropout_rate, pool_ratio=self.pool_ratio,
                    lamb=self.lamb, sample='true', agg_mode=self.agg_mode).to(DEVICE)

        elif self.layer_type == 'MRConv':
            self.high_model = Model_MRConv_LSTM(in_feat= HIGH_FEAT_NUM + OUTPUT_FEAT_NUM, out_feat=128, hid_feat=32,
                    conv_layers=self.num_layers, dropout=self.dropout_rate, pool_ratio=self.pool_ratio,
                    lamb=self.lamb, sample='true', agg_mode=self.agg_mode).to(DEVICE)
            
    def compute_assigned_feats(self, graph, feats, assignment):
        """
        Use the assignment matrix to agg the feats
        """
        num_nodes_per_graph = graph.batch_num_nodes().tolist()
        num_nodes_per_graph.insert(0, 0)
        intervals = [sum(num_nodes_per_graph[:i + 1])
                    for i in range(len(num_nodes_per_graph))]

        ll_h_concat = []
        for i in range(1, len(intervals)):
            h_agg = torch.matmul(assignment[i - 1].to(DEVICE), feats[intervals[i - 1]:intervals[i], :])
            ll_h_concat.append(h_agg)

        return torch.cat(ll_h_concat, dim=0)

    def bivector_attention(self, high_vector, low_vector):

        att_low = self.att_conv_low(low_vector.unsqueeze(0).unsqueeze(0)).squeeze()
        att_high = self.att_conv_high(high_vector.unsqueeze(0).unsqueeze(0)).squeeze()
        
        if len(att_low.shape) == 1:
            att_low = att_low.unsqueeze(0)
            att_high = att_high.unsqueeze(0)

        fuse_att = att_low + att_high
        feat_s = self.att_conv1(fuse_att.unsqueeze(0).unsqueeze(0)).squeeze()


        if len(feat_s.shape) == 1:
            feat_s = feat_s.unsqueeze(0)
        a_b = self.fc2(feat_s.unsqueeze(0).unsqueeze(0)).squeeze()
        a_b = self.softmax(a_b)
        feat_low = att_low * a_b[0]
        feat_high = att_high * a_b[1]
        

        final_vector = feat_high + feat_low

        pred = self.MLP(final_vector)

        return pred

    def concat_classification(self,high_graph_vector, low_graph_vector):
        concat_vector = torch.cat((high_graph_vector, low_graph_vector), dim=1)

        logits = self.MLP(concat_vector)

        return logits
    
    def check_nan(self, tensor):
        """Check if a PyTorch tensor contains NaN values."""
        if torch.isnan(tensor).any():
            nan_indices = torch.isnan(tensor).nonzero()
            print("Warning: NaN values found at indices:", nan_indices)
            return True
        return False

    def forward(
        self,
        low_graph: list,
        high_graph: list,
        ass_matrix: torch.Tensor
    ):
        updated_low_graph = self.pre_model(low_graph,low_graph.ndata['feat'])
        updated_feats = updated_low_graph.ndata['feat']
        low_graph_vector = self.low_model(updated_low_graph, updated_low_graph.ndata['feat'])

        if self.ass:
            low_concat = self.compute_assigned_feats(
                low_graph, updated_feats, ass_matrix)

            high_graph.ndata['feat'] = torch.cat(
                (low_concat, high_graph.ndata['feat']), dim=1)
        

        high_graph_vector = self.high_model(high_graph,high_graph.ndata['feat'])

        if self.mod == 'all':
            if self.fusion_mode == 'AFM':
                final_vector = self.bivector_attention(high_graph_vector, low_graph_vector)
            elif self.fusion_mode == 'sum':
                final_vector = torch.sum(torch.stack((high_graph_vector, low_graph_vector)),dim=0)
                final_vector = self.MLP(final_vector)
            elif self.fusion_mode == 'mean':
                final_vector = torch.mean(torch.stack((high_graph_vector, low_graph_vector)),dim=0)
                final_vector = self.MLP(final_vector)
            elif self.fusion_mode == 'concat':
                final_vector = self.concat_classification(high_graph_vector, low_graph_vector)
        elif self.mod == 'high':
            final_vector = self.MLP(high_graph_vector)
        elif self.mod == 'low':
            final_vector = self.MLP(low_graph_vector)
        return final_vector


def set_graph_on_cuda(graph, device):
    graph = graph.to(device)
    for key, val in graph.ndata.items():
        graph.ndata[key] = val.to(device)
    for key, val in graph.edata.items():
        graph.edata[key] = val.to(device)
    return graph


def ensure_requires_grad(graph):

    if not graph.batch_size:
        feat = graph.nodes['_U'].data['feat']
        if not feat.requires_grad:
            feat.requires_grad_(True)
            graph = dgl.unbatch(graph)[0]  
            graph = dgl.add_reverse_edges(graph)
            graph = dgl.batch([graph])
    else:

        sub_graphs = []
        for g in dgl.unbatch(graph):
            feat = g.nodes['_U'].data['feat']
            if not feat.requires_grad:
                feat.requires_grad_(True)
                g = dgl.add_reverse_edges(g)
            sub_graphs.append(g)
        graph = dgl.batch(sub_graphs)

    assert graph.batch_size > 0 or graph.nodes['_U'].data['feat'].requires_grad, "图结点特征未开启 requires_grad"
    for ntype in graph.ntypes:
        assert graph.nodes[ntype].data.get(
            'feat') is None or graph.nodes[ntype].data['feat'].requires_grad, f"{ntype} 结点特征未开启 requires_grad"

    return graph


def plot_confusion_matrix(cm, save_path):

    class_names = [k for k, v in sorted(DISASTER_TYPE.items(), key=lambda item: item[1])]

    plt.figure(figsize=(10, 8))
    

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.savefig(save_path + '/' + 'confusion_matrix.png')

    plt.close()

def calculate_metrics(cm):

    n_classes = cm.shape[0]
    metrics = {}
    
    for i in range(n_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        
        metrics[f'class_{i}'] = {
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN
        }
    
    return metrics 
