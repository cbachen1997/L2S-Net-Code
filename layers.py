import logging

import dgl
import dgl.function as fn
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import AvgPooling, GraphConv, MaxPooling
from dgl.ops import edge_softmax
from torch import Tensor
from torch.nn.parameter import Parameter
import torch_scatter
import scipy.sparse
from dgl.nn.pytorch import GATConv, GATv2Conv, GINConv, SAGEConv, PNAConv, EdgeConv
from torch.nn import Sequential as Seq, Linear as Lin
import numpy as np

from functions import edge_sparsemax
from utils4pre import get_batch_id, topk

def scatter_(name, src, index, dim=0, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index. (default: :obj:`0`)
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    assert name in ['add', 'mean', 'min', 'max']

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    out = op(src, index, dim, None, dim_size)
    out = out[0] if isinstance(out, tuple) else out

    if name == 'max':
        out[out < -10000] = 0
    elif name == 'min':
        out[out > 10000] = 0

    return out


class MaxRelativeGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='max'):
        super(MaxRelativeGraphConv, self).__init__()
        self.nn = MLP([in_channels*2, out_channels], act, norm, bias)
        self.aggr = aggr

    def forward(self, graph: DGLGraph, n_feat):
        edge_index = graph.edges()

        x_i = torch.index_select(n_feat, 0, edge_index[0])
        x_j = torch.index_select(n_feat, 0, edge_index[1])
        relative_feature_diff = x_i - x_j
        x_j = scatter_(self.aggr, relative_feature_diff, edge_index[1], dim_size=graph.num_nodes())

        new_features = torch.cat([n_feat, x_j], dim=1)

        return self.nn(new_features)


class WeightedGraphConv(GraphConv):
    r"""
    Description
    -----------
    GraphConv with edge weights on homogeneous graphs.
    If edge weights are not given, directly call GraphConv instead.

    Parameters
    ----------
    graph : DGLGraph
        The graph to perform this operation.
    n_feat : torch.Tensor
        The node features
    e_feat : torch.Tensor, optional
        The edge features. Default: :obj:`None`
    """
    def forward(self, graph:DGLGraph, n_feat, e_feat=None):
        if e_feat is None:
            return super(WeightedGraphConv, self).forward(graph, n_feat)
        
        with graph.local_scope():
            if self.weight is not None:
                n_feat = torch.matmul(n_feat, self.weight)
            src_norm = torch.pow(graph.out_degrees().float().clamp(min=1), -0.5)
            src_norm = src_norm.view(-1, 1)
            dst_norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
            dst_norm = dst_norm.view(-1, 1)
            n_feat = n_feat * src_norm
            graph.ndata["h"] = n_feat
            graph.edata["e"] = e_feat
            graph.update_all(fn.src_mul_edge("h", "e", "m"),
                            fn.sum("m", "h"))
            n_feat = graph.ndata.pop("h")
            n_feat = n_feat * dst_norm
            if self.bias is not None:
                n_feat = n_feat + self.bias
            if self._activation is not None:
                n_feat = self._activation(n_feat)
            return n_feat


class NodeInfoScoreLayer(nn.Module):
    r"""
    Description
    -----------
    Compute a score for each node for sort-pooling. The score of each node
    is computed via the absolute difference of its first-order random walk
    result and its features.

    Arguments
    ---------
    sym_norm : bool, optional
        If true, use symmetric norm for adjacency.
        Default: :obj:`True`

    Parameters
    ----------
    graph : DGLGraph
        The graph to perform this operation.
    feat : torch.Tensor
        The node features
    e_feat : torch.Tensor, optional
        The edge features. Default: :obj:`None`
    
    Returns
    -------
    Tensor
        Score for each node.
    """
    def __init__(self, sym_norm:bool=True):
        super(NodeInfoScoreLayer, self).__init__()
        self.sym_norm = sym_norm

    def forward(self, graph:dgl.DGLGraph, feat:Tensor, e_feat:Tensor):
            if self.sym_norm:

                src_norm = torch.pow(graph.out_degrees().float().clamp(min=1), -0.5)
                src_norm = src_norm.view(-1, 1).to(feat.device)

                dst_norm = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5)
                dst_norm = dst_norm.view(-1, 1).to(feat.device)


                src_feat = feat * src_norm
                

                graph.ndata["h"] = src_feat
                graph.edata["e"] = e_feat

                graph = dgl.remove_self_loop(graph)

                graph.update_all(fn.src_mul_edge("h", "e", "m"), fn.sum("m", "h"))
                

                dst_feat = graph.ndata.pop("h") * dst_norm
                feat = feat - dst_feat
            else:
                dst_norm = 1. / graph.in_degrees().float().clamp(min=1)
                dst_norm = dst_norm.view(-1, 1)

                graph.ndata["h"] = feat
                graph.edata["e"] = e_feat
                graph = dgl.remove_self_loop(graph)
                graph.update_all(fn.src_mul_edge("h", "e", "m"), fn.sum("m", "h"))

                feat = feat - dst_norm * graph.ndata.pop("h")

            score = torch.sum(torch.abs(feat), dim=1)
            return score


class HGPSLPool(nn.Module):
    r"""

    Description
    -----------
    The HGP-SL pooling layer from 
    `Hierarchical Graph Pooling with Structure Learning <https://arxiv.org/pdf/1911.05954.pdf>`

    Parameters
    ----------
    in_feat : int
        The number of input node feature's channels
    ratio : float, optional
        Pooling ratio. Default: 0.8
    sample : bool, optional
        Whether use k-hop union graph to increase efficiency. 
        Currently we only support full graph. Default: :obj:`False`
    sym_score_norm : bool, optional
        Use symmetric norm for adjacency or not. Default: :obj:`True`
    sparse : bool, optional
        Use edge sparsemax instead of edge softmax. Default: :obj:`True`
    sl : bool, optional
        Use structure learining module or not. Default: :obj:`True`
    lamb : float, optional
        The lambda parameter as weight of raw adjacency as described in the
        HGP-SL paper. Default: 1.0
    negative_slop : float, optional
        Negative slop for leaky_relu. Default: 0.2
    
    Returns
    -------
    DGLGraph
        The pooled graph.
    torch.Tensor
        Node features
    torch.Tensor
        Edge features
    torch.Tensor
        Permutation index
    """
    def __init__(self, in_feat:int, ratio=0.8, sample=True, 
                 sym_score_norm=True, sparse=True, sl=True,
                 lamb=1.0, negative_slop=0.2, k_hop=3):
        super(HGPSLPool, self).__init__()
        self.in_feat = in_feat
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.lamb = lamb
        self.negative_slop = negative_slop
        self.k_hop = k_hop

        self.att = Parameter(torch.Tensor(1, self.in_feat * 2))
        self.calc_info_score = NodeInfoScoreLayer(sym_norm=sym_score_norm)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.att.data)

    def forward(self, graph:DGLGraph, feat:Tensor, e_feat=None):

        if e_feat is None:
            e_feat = torch.ones((graph.number_of_edges(),), 
                                dtype=feat.dtype, device=feat.device)
        batch_num_nodes = graph.batch_num_nodes()

        x_score = self.calc_info_score(graph, feat, e_feat)
        perm, next_batch_num_nodes = topk(x_score, self.ratio, 
                                          get_batch_id(batch_num_nodes),
                                          batch_num_nodes)
        

        feat = feat[perm]
        pool_graph = None
        if not self.sample or not self.sl:
            graph.edata["e"] = e_feat
            pool_graph = dgl.node_subgraph(graph, perm)
            e_feat = pool_graph.edata.pop("e")
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)

        if not self.sl:
            return pool_graph, feat, e_feat, perm


        if self.sample:

            row, col = graph.all_edges()
            num_nodes = graph.num_nodes()

            scipy_adj = scipy.sparse.coo_matrix((e_feat.detach().cpu(), (row.detach().cpu(), col.detach().cpu())), shape=(num_nodes, num_nodes))
            for _ in range(self.k_hop):
                two_hop = scipy_adj ** 2
                two_hop = two_hop * (1e-5 / two_hop.max())
                scipy_adj = two_hop + scipy_adj
            row, col = scipy_adj.nonzero()
            row = torch.tensor(row, dtype=torch.long, device=graph.device)
            col = torch.tensor(col, dtype=torch.long, device=graph.device)
            e_feat = torch.tensor(scipy_adj.data, dtype=torch.float, device=feat.device)


            mask = perm.new_full((num_nodes, ), -1)
            i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
            mask[perm] = i
            row, col = mask[row], mask[col]
            mask = (row >=0 ) & (col >= 0)
            row, col = row[mask], col[mask]
            e_feat = e_feat[mask]


            mask = row != col
            num_nodes = perm.size(0) 
            loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
            inv_mask = ~mask
            loop_weight = torch.full((num_nodes, ), 0, dtype=e_feat.dtype, device=e_feat.device)
            remaining_e_feat = e_feat[inv_mask]
            if remaining_e_feat.numel() > 0:
                loop_weight[row[inv_mask]] = remaining_e_feat
            e_feat = torch.cat([e_feat[mask], loop_weight], dim=0)
            row, col = row[mask], col[mask]
            row = torch.cat([row, loop_index], dim=0)
            col = torch.cat([col, loop_index], dim=0)


            weights = (torch.cat([feat[row], feat[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop) + e_feat * self.lamb
            

            sl_graph = dgl.graph((row, col))
            if self.sparse:
                weights = edge_sparsemax(sl_graph, weights)
            else:
                weights = edge_softmax(sl_graph, weights)
            

            mask = torch.abs(weights) > 0
            row, col, weights = row[mask], col[mask], weights[mask]
            pool_graph = dgl.graph((row, col))
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)
            e_feat = weights

        else: 

            batch_num_nodes = next_batch_num_nodes
            block_begin_idx = torch.cat([batch_num_nodes.new_zeros(1), 
                batch_num_nodes.cumsum(dim=0)[:-1]], dim=0)
            block_end_idx = batch_num_nodes.cumsum(dim=0)
            dense_adj = torch.zeros((pool_graph.num_nodes(),
                                    pool_graph.num_nodes()),
                                    dtype=torch.float, 
                                    device=feat.device)
            for idx_b, idx_e in zip(block_begin_idx, block_end_idx):
                dense_adj[idx_b:idx_e, idx_b:idx_e] = 1.
            row, col = torch.nonzero(dense_adj).t().contiguous()


            weights = (torch.cat([feat[row], feat[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop)
            dense_adj[row, col] = weights

            pool_row, pool_col = pool_graph.all_edges()
            dense_adj[pool_row, pool_col] += self.lamb * e_feat
            weights = dense_adj[row, col]
            del dense_adj
            torch.cuda.empty_cache()

            complete_graph = dgl.graph((row, col))
            if self.sparse:
                weights = edge_sparsemax(complete_graph, weights)
            else:
                weights = edge_softmax(complete_graph, weights)

            mask = torch.abs(weights) > 1e-9
            row, col, weights = row[mask], col[mask], weights[mask]
            e_feat = weights
            pool_graph = dgl.graph((row, col))
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)

        return pool_graph, feat, e_feat, perm
    
class HGPSLPool_GAT(nn.Module):
    r"""

    Description
    -----------
    The HGP-SL pooling layer from 
    `Hierarchical Graph Pooling with Structure Learning <https://arxiv.org/pdf/1911.05954.pdf>`

    Parameters
    ----------
    in_feat : int
        The number of input node feature's channels
    ratio : float, optional
        Pooling ratio. Default: 0.8
    sample : bool, optional
        Whether use k-hop union graph to increase efficiency. 
        Currently we only support full graph. Default: :obj:`False`
    sym_score_norm : bool, optional
        Use symmetric norm for adjacency or not. Default: :obj:`True`
    sparse : bool, optional
        Use edge sparsemax instead of edge softmax. Default: :obj:`True`
    sl : bool, optional
        Use structure learining module or not. Default: :obj:`True`
    lamb : float, optional
        The lambda parameter as weight of raw adjacency as described in the
        HGP-SL paper. Default: 1.0
    negative_slop : float, optional
        Negative slop for leaky_relu. Default: 0.2
    
    Returns
    -------
    DGLGraph
        The pooled graph.
    torch.Tensor
        Node features
    torch.Tensor
        Edge features
    torch.Tensor
        Permutation index
    """
    def __init__(self, in_feat:int, ratio=0.8, sample=True, 
                 sym_score_norm=True, sparse=True, sl=True,
                 lamb=1.0, negative_slop=0.2, k_hop=3):
        super(HGPSLPool_GAT, self).__init__()
        self.in_feat = in_feat
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.lamb = lamb
        self.negative_slop = negative_slop
        self.k_hop = k_hop

        self.att = Parameter(torch.Tensor(1, self.in_feat * 2))
        self.calc_info_score = NodeInfoScoreLayer(sym_norm=sym_score_norm)

        self.linear = torch.nn.Linear(in_feat*4, in_feat)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.att.data)

    def forward(self, graph:DGLGraph, feat:Tensor, e_feat=None):

        if e_feat is None:

            e_feat = torch.ones((graph.number_of_edges(),), 
                                dtype=feat.dtype, device=feat.device)
        batch_num_nodes = graph.batch_num_nodes()

        feat_concat = torch.reshape(feat, (feat.shape[0], self.in_feat*4))
        feat = self.linear(feat_concat)

        x_score = self.calc_info_score(graph, feat, e_feat)

        perm, next_batch_num_nodes = topk(x_score, self.ratio, 
                                          get_batch_id(batch_num_nodes),
                                          batch_num_nodes)
        

        feat = feat[perm]
        pool_graph = None
        if not self.sample or not self.sl:

            graph.edata["e"] = e_feat
            pool_graph = dgl.node_subgraph(graph, perm)
            e_feat = pool_graph.edata.pop("e")
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)

        if not self.sl:
            return pool_graph, feat, e_feat, perm


        if self.sample:

            row, col = graph.all_edges()
            num_nodes = graph.num_nodes()


            scipy_adj = scipy.sparse.coo_matrix((e_feat.detach().cpu(), (row.detach().cpu(), col.detach().cpu())), shape=(num_nodes, num_nodes))
            for _ in range(self.k_hop):
                two_hop = scipy_adj ** 2
                two_hop = two_hop * (1e-5 / two_hop.max())
                scipy_adj = two_hop + scipy_adj

            row, col = scipy_adj.nonzero()
            row = torch.tensor(row, dtype=torch.long, device=graph.device)
            col = torch.tensor(col, dtype=torch.long, device=graph.device)
            e_feat = torch.tensor(scipy_adj.data, dtype=torch.float, device=feat.device)

            mask = perm.new_full((num_nodes, ), -1)
            i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
            mask[perm] = i
            row, col = mask[row], mask[col]
            mask = (row >=0 ) & (col >= 0)
            row, col = row[mask], col[mask]
            e_feat = e_feat[mask]


            mask = row != col
            num_nodes = perm.size(0)
            loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
            inv_mask = ~mask
            loop_weight = torch.full((num_nodes, ), 0, dtype=e_feat.dtype, device=e_feat.device)
            remaining_e_feat = e_feat[inv_mask]
            if remaining_e_feat.numel() > 0:
                loop_weight[row[inv_mask]] = remaining_e_feat
            e_feat = torch.cat([e_feat[mask], loop_weight], dim=0)
            row, col = row[mask], col[mask]
            row = torch.cat([row, loop_index], dim=0)
            col = torch.cat([col, loop_index], dim=0)


            weights = (torch.cat([feat[row], feat[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop) + e_feat * self.lamb
            

            sl_graph = dgl.graph((row, col))
            if self.sparse:
                weights = edge_sparsemax(sl_graph, weights)
            else:
                weights = edge_softmax(sl_graph, weights)

            mask = torch.abs(weights) > 0
            row, col, weights = row[mask], col[mask], weights[mask]
            pool_graph = dgl.graph((row, col))
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)
            e_feat = weights

        else: 

            batch_num_nodes = next_batch_num_nodes
            block_begin_idx = torch.cat([batch_num_nodes.new_zeros(1), 
                batch_num_nodes.cumsum(dim=0)[:-1]], dim=0)
            block_end_idx = batch_num_nodes.cumsum(dim=0)
            dense_adj = torch.zeros((pool_graph.num_nodes(),
                                    pool_graph.num_nodes()),
                                    dtype=torch.float, 
                                    device=feat.device)
            for idx_b, idx_e in zip(block_begin_idx, block_end_idx):
                dense_adj[idx_b:idx_e, idx_b:idx_e] = 1.
            row, col = torch.nonzero(dense_adj).t().contiguous()


            weights = (torch.cat([feat[row], feat[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop)
            dense_adj[row, col] = weights


            pool_row, pool_col = pool_graph.all_edges()
            dense_adj[pool_row, pool_col] += self.lamb * e_feat
            weights = dense_adj[row, col]
            del dense_adj
            torch.cuda.empty_cache()


            complete_graph = dgl.graph((row, col))
            if self.sparse:
                weights = edge_sparsemax(complete_graph, weights)
            else:
                weights = edge_softmax(complete_graph, weights)

            mask = torch.abs(weights) > 1e-9
            row, col, weights = row[mask], col[mask], weights[mask]
            e_feat = weights
            pool_graph = dgl.graph((row, col))
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)

        return pool_graph, feat, e_feat, perm

class HGPSLPool_GIN(nn.Module):
    r"""

    Description
    -----------
    The HGP-SL pooling layer from 
    `Hierarchical Graph Pooling with Structure Learning <https://arxiv.org/pdf/1911.05954.pdf>`

    Parameters
    ----------
    in_feat : int
        The number of input node feature's channels
    ratio : float, optional
        Pooling ratio. Default: 0.8
    sample : bool, optional
        Whether use k-hop union graph to increase efficiency. 
        Currently we only support full graph. Default: :obj:`False`
    sym_score_norm : bool, optional
        Use symmetric norm for adjacency or not. Default: :obj:`True`
    sparse : bool, optional
        Use edge sparsemax instead of edge softmax. Default: :obj:`True`
    sl : bool, optional
        Use structure learining module or not. Default: :obj:`True`
    lamb : float, optional
        The lambda parameter as weight of raw adjacency as described in the
        HGP-SL paper. Default: 1.0
    negative_slop : float, optional
        Negative slop for leaky_relu. Default: 0.2
    
    Returns
    -------
    DGLGraph
        The pooled graph.
    torch.Tensor
        Node features
    torch.Tensor
        Edge features
    torch.Tensor
        Permutation index
    """
    def __init__(self, in_feat:int, ratio=0.8, sample=True, 
                 sym_score_norm=True, sparse=True, sl=True,
                 lamb=1.0, negative_slop=0.2, k_hop=3):
        super(HGPSLPool_GIN, self).__init__()
        self.in_feat = in_feat
        self.ratio = ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.lamb = lamb
        self.negative_slop = negative_slop
        self.k_hop = k_hop

        self.att = Parameter(torch.Tensor(1, self.in_feat * 2))
        self.calc_info_score = NodeInfoScoreLayer(sym_norm=sym_score_norm)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.att.data)

    def forward(self, graph:DGLGraph, feat:Tensor, e_feat=None):

        if e_feat is None:

            e_feat = torch.ones((graph.number_of_edges(),), 
                                dtype=feat.dtype, device=feat.device)
        batch_num_nodes = graph.batch_num_nodes()
        

        x_score = self.calc_info_score(graph, feat, e_feat)

        perm, next_batch_num_nodes = topk(x_score, self.ratio, 
                                          get_batch_id(batch_num_nodes),
                                          batch_num_nodes)

        feat = feat[perm]
        pool_graph = None
        if not self.sample or not self.sl:

            graph.edata["e"] = e_feat
            pool_graph = dgl.node_subgraph(graph, perm)
            e_feat = pool_graph.edata.pop("e")
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)


        if not self.sl:
            return pool_graph, feat, e_feat, perm

        if self.sample:

            row, col = graph.all_edges()
            num_nodes = graph.num_nodes()

            scipy_adj = scipy.sparse.coo_matrix((e_feat.detach().cpu(), (row.detach().cpu(), col.detach().cpu())), shape=(num_nodes, num_nodes))
            for _ in range(self.k_hop):
                two_hop = scipy_adj ** 2
                two_hop = two_hop * (1e-5 / two_hop.max())
                scipy_adj = two_hop + scipy_adj
            row, col = scipy_adj.nonzero()
            row = torch.tensor(row, dtype=torch.long, device=graph.device)
            col = torch.tensor(col, dtype=torch.long, device=graph.device)
            e_feat = torch.tensor(scipy_adj.data, dtype=torch.float, device=feat.device)


            mask = perm.new_full((num_nodes, ), -1)
            i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
            mask[perm] = i
            row, col = mask[row], mask[col]
            mask = (row >=0 ) & (col >= 0)
            row, col = row[mask], col[mask]
            e_feat = e_feat[mask]


            mask = row != col
            num_nodes = perm.size(0) 
            loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
            inv_mask = ~mask
            loop_weight = torch.full((num_nodes, ), 0, dtype=e_feat.dtype, device=e_feat.device)
            remaining_e_feat = e_feat[inv_mask]
            if remaining_e_feat.numel() > 0:
                loop_weight[row[inv_mask]] = remaining_e_feat
            e_feat = torch.cat([e_feat[mask], loop_weight], dim=0)
            row, col = row[mask], col[mask]
            row = torch.cat([row, loop_index], dim=0)
            col = torch.cat([col, loop_index], dim=0)


            weights = (torch.cat([feat[row], feat[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop) + e_feat * self.lamb
            

            sl_graph = dgl.graph((row, col))
            if self.sparse:
                weights = edge_sparsemax(sl_graph, weights)
            else:
                weights = edge_softmax(sl_graph, weights)
            

            mask = torch.abs(weights) > 0
            row, col, weights = row[mask], col[mask], weights[mask]
            pool_graph = dgl.graph((row, col))
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)
            e_feat = weights

        else: 

            batch_num_nodes = next_batch_num_nodes
            block_begin_idx = torch.cat([batch_num_nodes.new_zeros(1), 
                batch_num_nodes.cumsum(dim=0)[:-1]], dim=0)
            block_end_idx = batch_num_nodes.cumsum(dim=0)
            dense_adj = torch.zeros((pool_graph.num_nodes(),
                                    pool_graph.num_nodes()),
                                    dtype=torch.float, 
                                    device=feat.device)
            for idx_b, idx_e in zip(block_begin_idx, block_end_idx):
                dense_adj[idx_b:idx_e, idx_b:idx_e] = 1.
            row, col = torch.nonzero(dense_adj).t().contiguous()


            weights = (torch.cat([feat[row], feat[col]], dim=1) * self.att).sum(dim=-1)
            weights = F.leaky_relu(weights, self.negative_slop)
            dense_adj[row, col] = weights

            pool_row, pool_col = pool_graph.all_edges()
            dense_adj[pool_row, pool_col] += self.lamb * e_feat
            weights = dense_adj[row, col]
            del dense_adj
            torch.cuda.empty_cache()

            complete_graph = dgl.graph((row, col))
            if self.sparse:
                weights = edge_sparsemax(complete_graph, weights)
            else:
                weights = edge_softmax(complete_graph, weights)


            mask = torch.abs(weights) > 1e-9
            row, col, weights = row[mask], col[mask], weights[mask]
            e_feat = weights
            pool_graph = dgl.graph((row, col))
            pool_graph.set_batch_num_nodes(next_batch_num_nodes)

        return pool_graph, feat, e_feat, perm


class ConvPoolReadout(torch.nn.Module):
    """A helper class. (GraphConv -> Pooling -> Readout)"""
    def __init__(self, in_feat:int, out_feat:int, pool_ratio=0.8,
                 sample:bool=False, sparse:bool=True, sl:bool=True,
                 lamb:float=1., pool:bool=True):
        super(ConvPoolReadout, self).__init__()
        self.use_pool = pool
        self.conv = WeightedGraphConv(in_feat, out_feat)
        if pool:
            self.pool = HGPSLPool(out_feat, ratio=pool_ratio, sparse=sparse,
                                  sample=sample, sl=sl, lamb=lamb)
        else:
            self.pool = None
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def forward(self, graph, feature, e_feat=None):
        out = F.relu(self.conv(graph, feature, e_feat))
        if self.use_pool:
            graph, out, e_feat, _ = self.pool(graph, out, e_feat)
        readout = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)
        return graph, out, e_feat, readout
    
class ConvPoolReadout_GATv2(torch.nn.Module):
    """A helper class. (GraphConv -> Pooling -> Readout)"""
    def __init__(self, in_feat:int, out_feat:int, pool_ratio=0.8,
                 sample:bool=False, sparse:bool=True, sl:bool=True,
                 lamb:float=1., pool:bool=True, num_heads:int=4):
        super(ConvPoolReadout_GATv2, self).__init__()
        self.use_pool = pool
        self.conv = GATv2Conv(in_feat, out_feat, num_heads=num_heads)
        if pool:
            self.pool = HGPSLPool_GAT(out_feat, ratio=pool_ratio, sparse=sparse,
                                  sample=sample, sl=sl, lamb=lamb)
        else:
            self.pool = None
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def forward(self, graph, feature, e_feat=None):


        out = F.relu(self.conv(graph, feature))
        if self.use_pool:
            graph, out, e_feat, _ = self.pool(graph, out, e_feat)
        readout = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)

        return graph, out, e_feat, readout
    
class ConvPoolReadout_GAT(torch.nn.Module):
    """A helper class. (GraphConv -> Pooling -> Readout)"""
    def __init__(self, in_feat:int, out_feat:int, pool_ratio=0.8,
                 sample:bool=False, sparse:bool=True, sl:bool=True,
                 lamb:float=1., pool:bool=True, num_heads:int=4):
        super(ConvPoolReadout_GAT, self).__init__()
        self.use_pool = pool
        self.conv = GATConv(in_feat, out_feat, num_heads=num_heads)
        if pool:
            self.pool = HGPSLPool_GAT(out_feat, ratio=pool_ratio, sparse=sparse,
                                  sample=sample, sl=sl, lamb=lamb)
        else:
            self.pool = None
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def forward(self, graph, feature, e_feat=None):


        out = F.relu(self.conv(graph, feature))
        if self.use_pool:
            graph, out, e_feat, _ = self.pool(graph, out, e_feat)
        readout = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)

        return graph, out, e_feat, readout

    
class ConvPoolReadout_GIN(torch.nn.Module):
    """A helper class. (GraphConv -> Pooling -> Readout)"""
    def __init__(self, in_feat:int, out_feat:int, pool_ratio=0.8,
                 sample:bool=False, sparse:bool=True, sl:bool=True,
                 lamb:float=1., pool:bool=True):
        super(ConvPoolReadout_GIN, self).__init__()
        self.use_pool = pool
        self.conv = GINConv(torch.nn.Linear(in_feat, out_feat),aggregator_type = 'max')
        if pool:
            self.pool = HGPSLPool_GIN(out_feat, ratio=pool_ratio, sparse=sparse,
                                  sample=sample, sl=sl, lamb=lamb)
        else:
            self.pool = None
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def forward(self, graph, feature, e_feat=None):

        out = F.relu(self.conv(graph, feature))
        if self.use_pool:
            graph, out, e_feat, _ = self.pool(graph, out, e_feat)

        readout = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)

        return graph, out, e_feat, readout

class ConvPoolReadout_SAGE(torch.nn.Module):
    """A helper class. (GraphConv -> Pooling -> Readout)"""
    def __init__(self, in_feat:int, out_feat:int, pool_ratio=0.8,
                 sample:bool=False, sparse:bool=True, sl:bool=True,
                 lamb:float=1., pool:bool=True):
        super(ConvPoolReadout_SAGE, self).__init__()
        self.use_pool = pool

        self.conv = SAGEConv(in_feat, out_feat, aggregators='mean')
        if pool:

            self.pool = HGPSLPool_GIN(out_feat, ratio=pool_ratio, sparse=sparse,
                                  sample=sample, sl=sl, lamb=lamb)
        else:
            self.pool = None
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def forward(self, graph, feature, e_feat=None):


        out = F.relu(self.conv(graph, feature))
        if self.use_pool:
            graph, out, e_feat, _ = self.pool(graph, out, e_feat)

        readout = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)

        return graph, out, e_feat, readout
    
class ConvPoolReadout_PNA(torch.nn.Module):
    """A helper class. (GraphConv -> Pooling -> Readout)"""
    def __init__(self, in_feat:int, out_feat:int, pool_ratio=0.8,
                 sample:bool=False, sparse:bool=True, sl:bool=True,
                 lamb:float=1., pool:bool=True):
        super(ConvPoolReadout_PNA, self).__init__()
        self.use_pool = pool

        self.aggregator = ['max','min','std','var'] 
        self.scalers = ['identity', 'amplification','attenuation']
        self.deg = 4

        self.conv = PNAConv(in_feat, out_feat, aggregators=self.aggregator, scalers=self.scalers, delta=self.deg)
        if pool:

            self.pool = HGPSLPool_GIN(out_feat, ratio=pool_ratio, sparse=sparse,
                                  sample=sample, sl=sl, lamb=lamb)
        else:
            self.pool = None
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def forward(self, graph, feature, e_feat=None):


        out = F.relu(self.conv(graph, feature))
        if self.use_pool:
            graph, out, e_feat, _ = self.pool(graph, out, e_feat)

        readout = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)

        return graph, out, e_feat, readout
    
class ConvPoolReadout_EdgeConv(torch.nn.Module):
    """A helper class. (GraphConv -> Pooling -> Readout)"""
    def __init__(self, in_feat:int, out_feat:int, pool_ratio=0.8,
                 sample:bool=False, sparse:bool=True, sl:bool=True,
                 lamb:float=1., pool:bool=True):
        super(ConvPoolReadout_EdgeConv, self).__init__()
        self.use_pool = pool

        self.conv = MaxRelativeGraphConv(in_feat, out_feat)
        if pool:

            self.pool = HGPSLPool_GIN(out_feat, ratio=pool_ratio, sparse=sparse,
                                  sample=sample, sl=sl, lamb=lamb)
        else:
            self.pool = None
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def forward(self, graph, feature, e_feat=None):


        out = F.relu(self.conv(graph, feature))
        if self.use_pool:
            graph, out, e_feat, _ = self.pool(graph, out, e_feat)

        readout = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)

        return graph, out, e_feat, readout
    
class ConvPoolReadout_MRConv(torch.nn.Module):
    """A helper class. (GraphConv -> Pooling -> Readout)"""
    def __init__(self, in_feat:int, out_feat:int, pool_ratio=0.8,
                 sample:bool=False, sparse:bool=True, sl:bool=True,
                 lamb:float=1., pool:bool=True):
        super(ConvPoolReadout_MRConv, self).__init__()
        self.use_pool = pool

        self.conv = EdgeConv(in_feat, out_feat)
        if pool:

            self.pool = HGPSLPool_GIN(out_feat, ratio=pool_ratio, sparse=sparse,
                                  sample=sample, sl=sl, lamb=lamb)
        else:
            self.pool = None
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

    def forward(self, graph, feature, e_feat=None):


        out = F.relu(self.conv(graph, feature))
        if self.use_pool:
            graph, out, e_feat, _ = self.pool(graph, out, e_feat)

        readout = torch.cat([self.avgpool(graph, out), self.maxpool(graph, out)], dim=-1)

        return graph, out, e_feat, readout