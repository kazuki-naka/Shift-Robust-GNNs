from dgl.nn.pytorch.conv import GATConv
import torch as th
from torch import nn

import loralib as lora

class GATConv(GATConv): 
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
        finetune = False
    ):
        super(GATConv, self).__init__()
        if isinstance(in_feats, tuple):
            if finetune: 
                self.fc_src = lora.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=False
                )
                self.fc_dst = lora.Linear(
                    self._in_dst_feats, out_feats * num_heads, bias=False
                )
            else: 
                self.fc_src = nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=False
                )
                self.fc_dst = nn.Linear(
                    self._in_dst_feats, out_feats * num_heads, bias=False
                )
        else:
            if finetune: 
                self.fc = lora.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=False
                )
            else: 
                self.fc = nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=False
                )
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
