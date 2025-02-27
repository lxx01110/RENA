from typing import Optional
from itertools import chain

import torch
import torch.nn as nn

from .gat import GAT

from utils import create_norm, drop_edge


def setup_module(enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:

    mod = GAT(
        in_dim=in_dim,
        num_hidden=num_hidden,
        out_dim=out_dim,
        num_layers=num_layers,
        nhead=nhead,
        nhead_out=nhead_out,
        concat_out=concat_out,
        activation=activation,
        feat_drop=dropout,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        norm=create_norm(norm),
        encoding=(enc_dec == "encoding"),
    )
  
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            features_unmask,
            observed,
            imputed,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            drop_edge_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        self.feature_unmask = features_unmask
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
       
        self.observed = observed
        self.imputed = imputed
        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        enc_num_hidden = num_hidden // nhead
        enc_nhead = nhead
        

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden 
        self.imp_feat = nn.Parameter(torch.empty(size=(features_unmask.size(0), in_dim)))
        nn.init.xavier_normal_(self.imp_feat.data, gain=1.414)
        self.encoder = setup_module(
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

         
        self.decoder = setup_module(
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = nn.MSELoss()

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def forward(self, g, x):
        loss = self.mask_attr_prediction(g, x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item
    
    def mask_attr_prediction(self, g, x):
        
        feature = torch.where(torch.isnan(x), self.imp_feat, x)
        pre_use_g, use_x = g, feature
        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden = self.encoder(use_g, use_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type in ("mlp", "liear") :
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)
        
       

        x_init_reliable = x[self.feature_unmask]
        x_rec_reliable = recon[self.feature_unmask]

        x_init_unreliable = x[~self.feature_unmask]
        x_rec_unreliable = recon[~self.feature_unmask]


        loss = self.observed * self.criterion(x_init_reliable, x_rec_reliable) + self.imputed * self.criterion(x_init_unreliable, x_rec_unreliable)
        return loss

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
