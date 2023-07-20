import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import model.operations as o
from model.genotypes import Genotype


class SearchCell(nn.Module):
    def __init__(self, num_nodes, multiplier, c_prev_prev, c_prev, c, reduce, reduce_prev, primitives, dropout_proba):
        super().__init__()

        self.num_nodes = num_nodes
        self.preproc0 = o.FactorizedReduce(c_prev_prev, c) if reduce_prev else o.ReLUConvBN(c_prev_prev, c, 1, 1, 0)
        self.preproc1 = o.ReLUConvBN(c_prev, c, 1, 1, 0)
        self.reduce, self.multiplier = reduce, multiplier
        self.dag = nn.ModuleList()

        for i in range(self.num_nodes):
            for j in range(2 + i):
                stride = 2 if (reduce and j < 2) else 1
                edge_op = o.MixedOp(c, stride, primitives, dropout_proba)
                self.dag.append(edge_op)

    def forward(self, s0, s1, weights):
        s0, s1 = self.preproc0(s0), self.preproc1(s1)
        states = [s0, s1]
        offset = 0

        for i in range(self.num_nodes):
            s = sum(self.dag[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
                
        return torch.cat(states[-self.multiplier:], dim=1)
    

class SearchNetwork(nn.Module):
    def __init__(
            self, c_in, c, num_classes, num_layers, primitives, num_nodes=4, multiplier=4,
            stem_multiplier=3, dropout_proba=0.0
    ):
        super().__init__()

        self.num_class = num_classes
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.primitives = primitives
        self.multiplier = multiplier
        self.stem_multiplier = stem_multiplier
        self.dropout_proba = dropout_proba
        self.cells = nn.ModuleList()
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, self.stem_multiplier * c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stem_multiplier * c)
        )
        # The i^th node has i connections from the preceding i nodes.
        # There are two additional connections from C_{k-1} and C_{k-2}.
        k = sum(i+2 for i in range(self.num_nodes))
        self.alpha_normal = nn.Parameter(1e-3 * torch.randn(k, len(self.primitives)))
        self.alpha_reduce = nn.Parameter(1e-3 * torch.randn(k, len(self.primitives)))

        c_out = self._compile_network(self.stem_multiplier * c, self.stem_multiplier * c, c, False)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_out, self.num_class)

        self.alphas = [self.alpha_normal, self.alpha_reduce]
        self.weights = [p for n, p in self.named_parameters() if 'alpha' not in n]

    def forward(self, x):
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1, weights_normal if cell.reduce else weights_reduce)

        out = self.global_pool(s1)
        out = out.reshape(out.size(0), -1)
        logits = self.classifier(out)

        return logits
    
    @property
    def genotype(self):
        gene_normal = self._parse_alpha(self.alpha_normal)
        gene_reduce = self._parse_alpha(self.alpha_reduce)
        concat = range(2, 2 + self.num_nodes)
        return Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
    
    @torch.no_grad()
    def reset_alphas(self):
        shape, device = self.alpha_normal.shape, self.alpha_normal.device
        self.alpha_normal.data = 1e-3 * torch.randn(shape).to(device)
        self.alpha_reduce.data = 1e-3 * torch.randn(shape).to(device)

    def _compile_network(self, c_prev_prev, c_prev, c_curr, reduce_prev):
        for i in range(self.num_layers):
            reduce = i in [self.num_layers // 3, 2 * self.num_layers // 3]
            c_curr = c_curr * 2 if reduce else c_curr
            cell = SearchCell(
                self.num_nodes, self.multiplier, c_prev_prev, c_prev, c_curr, reduce, reduce_prev,
                self.primitives, self.dropout_proba
            )
            self.cells.append(cell)

            c_curr_out = c_curr * self.num_nodes
            c_prev_prev, c_prev = c_prev, c_curr_out
            reduce_prev = reduce

        return c_prev

    def _parse_alpha(self, weights, k=2):
        sizes = [i+2 for i in range(self.num_nodes)]
        alpha = np.split(weights, np.cumsum(sizes)[:-1])
        none_idx = self.primitives.index('none') if 'none' in self.primitives else -1
        if none_idx >= 0:
            mask = np.ones(len(self.primitives), dtype=bool)
            mask[none_idx] = False
            alpha = [a[:, mask] for a in alpha]

        genes = []
        for edges in alpha:
            edge_max, primitive_indices = torch.topk(edges, 1)
            topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
            node_gene = []
            for edge_idx in topk_edge_indices:
                prim_idx = primitive_indices[edge_idx]
                prim = self.primitives[prim_idx]
                node_gene.append((prim, edge_idx.item()))
            genes.extend(node_gene)
        
        return genes
