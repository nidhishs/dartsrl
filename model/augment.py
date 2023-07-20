import torch
import torch.nn as nn

import model.operations as o


class AugmentCell(nn.Module):
    def __init__(self, genotype, c_prev_prev, c_prev, c, reduce, reduce_prev, dropout):
        super().__init__()

        self._preproc0 = (
            o.FactorizedReduce(c_prev_prev, c)
            if reduce_prev
            else o.ReLUConvBN(c_prev_prev, c, 1, 1, 0)
        )
        self._preproc1 = o.ReLUConvBN(c_prev, c, 1, 1, 0)
        self.reduce = reduce
        self.dropout = dropout

        if reduce:
            self._op_names, self._indices = zip(*genotype.reduce)
            self._concat = genotype.reduce_concat
        else:
            self._op_names, self._indices = zip(*genotype.normal)
            self._concat = genotype.normal_concat

        self.num_nodes = len(self._op_names) // 2
        self.multiplier = len(self._concat)
        self._ops = nn.ModuleList()
        self._compile_cell(c)

    def _compile_cell(self, c):
        for name, index in zip(self._op_names, self._indices):
            stride = 2 if (self.reduce and index < 2) else 1
            op = o.OPERATIONS[name](c, stride, True)
            if "pool" in name:
                op = nn.Sequential(op, nn.BatchNorm2d(c, affine=False))
            if isinstance(op, nn.Identity) and self.dropout > 0:
                op = nn.Sequential(op, nn.Dropout(self.dropout))
            self._ops.append(op)

    def forward(self, s0, s1):
        s0, s1 = self._preproc0(s0), self._preproc1(s1)
        states = [s0, s1]

        for i in range(self.num_nodes):
            o1, o2 = self._ops[2 * i], self._ops[2 * i + 1]
            x1, x2 = states[self._indices[2 * i]], states[self._indices[2 * i + 1]]
            x1, x2 = o1(x1), o2(x2)
            s = x1 + x2
            states.append(s)

        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.ReLU(inplace=True), # image size = 8 x 8
            nn.AvgPool2d(
                5, stride=3, padding=0, count_include_pad=False
            ), # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):
    def __init__(self, c, num_classes, num_layers, auxiliary, genotype, dropout=0.2):
        super().__init__()

        self.num_layers = num_layers
        self.auxiliary = auxiliary
        self.genotype = genotype
        self.dropout = dropout

        stem_multiplier = 3
        c_curr = stem_multiplier * c
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_curr),
        )

        c_prev_prev, c_prev, c_curr = c_curr, c_curr, c
        self.cells = nn.ModuleList()
        c_out, c_to_auxiliary = self._compile_network(c_prev_prev, c_prev, c_curr, False)

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(c_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_out, num_classes)
    
    def _compile_network(self, c_prev_prev, c_prev, c_curr, reduce_prev):
        for i in range(self.num_layers):
            reduce = i in [self.num_layers // 3, 2 * self.num_layers // 3]
            c_curr = c_curr * 2 if reduce else c_curr
            cell = AugmentCell(self.genotype, c_prev_prev, c_prev, c_curr, reduce, reduce_prev, self.dropout)
            self.cells.append(cell)

            c_prev_prev, c_prev = c_prev, cell.multiplier * c_curr
            reduce_prev = reduce
            if i == 2 * self.num_layers // 3:
                c_to_auxiliary = c_prev
        
        return c_prev, c_to_auxiliary
    
    def forward(self, x):
        s0 = s1 = self.stem(x)

        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if self.auxiliary and self.training and i == (2 * self.num_layers // 3):
                logits_aux = self.auxiliary_head(s1)
        
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        if self.auxiliary and self.training:
            return logits, logits_aux
        else:
            return logits
        