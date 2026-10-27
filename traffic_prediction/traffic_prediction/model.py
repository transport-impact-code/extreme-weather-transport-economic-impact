import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A_norm):

        x = torch.matmul(A_norm, x)
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x


class STGNNWeatherSeq2Seq(nn.Module):

    def __init__(
        self,
        num_nodes,
        hist_dim=5,
        weather_dim=4,
        flow_dim=1,
        Tin=24,
        Tout=12,
        gcn_hidden=32,
        enc_hidden=128,
        dec_hidden=128,
        weather_hidden=32,
        dropout=0.1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.Tin = Tin
        self.Tout = Tout
        self.flow_dim = flow_dim

        # =================
        # Spatial GCN
        # =================

        self.gcn1 = GCNLayer(hist_dim, gcn_hidden, dropout)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden, dropout)

        self.ln_gcn = nn.LayerNorm(gcn_hidden)

        # =================
        # Encoder
        # =================

        self.enc_gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=enc_hidden,
            batch_first=True,
        )

        self.ln_ctx = nn.LayerNorm(enc_hidden)

        # =================
        # Decoder modules
        # =================

        self.flow_proj = nn.Sequential(
            nn.Linear(flow_dim, dec_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.weather_proj = nn.Sequential(
            nn.Linear(weather_dim, weather_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.ctx_to_dec = nn.Linear(enc_hidden, dec_hidden)

        self.dec_cell = nn.GRUCell(
            input_size=dec_hidden + weather_hidden + dec_hidden,
            hidden_size=dec_hidden,
        )

        self.out_head = nn.Linear(dec_hidden, flow_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_hist,
        x_future,
        A_norm,
        y_true=None,
        teacher_forcing_ratio=0.0,
    ):

        B, Tin, N, F = x_hist.shape

        # =================
        # Spatial GCN
        # =================

        gcn_seq = []

        for t in range(Tin):

            xt = x_hist[:, t]

            ht = self.gcn1(xt, A_norm)
            ht = self.gcn2(ht, A_norm)

            ht = self.ln_gcn(ht)

            gcn_seq.append(ht)

        gcn_out = torch.stack(gcn_seq, dim=1)

        # =================
        # Encoder
        # =================

        Hg = gcn_out.shape[-1]

        enc_in = (
            gcn_out
            .permute(0,2,1,3)
            .contiguous()
            .view(B*N, Tin, Hg)
        )

        _, h_last = self.enc_gru(enc_in)

        ctx = h_last.squeeze(0)

        ctx = self.ln_ctx(ctx)

        # =================
        # Decoder init
        # =================

        ctx_emb = torch.tanh(self.ctx_to_dec(ctx))

        h = ctx_emb

        prev_flow = x_hist[:, -1, :, 0:1].contiguous().view(B*N, self.flow_dim)

        outputs = []

        for k in range(self.Tout):

            flow_emb = self.flow_proj(prev_flow)

            weather_k = x_future[:, k].contiguous().view(B*N, -1)
            weather_emb = self.weather_proj(weather_k)

            dec_in = torch.cat(
                [flow_emb, weather_emb, ctx_emb],
                dim=-1
            )

            h = self.dec_cell(dec_in, h)

            h = self.dropout(h)

            pred = self.out_head(h)

            outputs.append(pred.view(B,N,self.flow_dim))

            if (y_true is not None) and (teacher_forcing_ratio>0):

                if torch.rand(1).item() < teacher_forcing_ratio:
                    prev_flow = y_true[:,k].contiguous().view(B*N,self.flow_dim)
                else:
                    prev_flow = pred

            else:

                prev_flow = pred

        y_hat = torch.stack(outputs, dim=1)

        return y_hat
