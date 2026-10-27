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

    def __init__(self, num_nodes, weather_force=5.0):
        super().__init__()

        self.weather_force = weather_force

        self.gcn1 = GCNLayer(5, 32)
        self.gcn2 = GCNLayer(32, 32)

        self.enc_gru = nn.GRU(32, 128, batch_first=True)

        self.flow_proj = nn.Linear(1, 128)
        self.weather_proj = nn.Linear(4, 32)

        self.dec_cell = nn.GRUCell(128 + 32 + 128, 128)

        self.out_head = nn.Linear(128, 1)

        self.weather_gate = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.weather_scale = nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        x_hist,
        x_future,
        A_norm,
        x_future_raw=None,
        y_true=None,
    ):

        B, Tin, N, _ = x_hist.shape

        gcn_seq = []

        for t in range(Tin):
            xt = x_hist[:, t]
            ht = self.gcn1(xt, A_norm)
            ht = self.gcn2(ht, A_norm)
            gcn_seq.append(ht)

        gcn_out = torch.stack(gcn_seq, dim=1)

        enc_in = gcn_out.permute(0,2,1,3).reshape(B*N, Tin, 32)

        _, h_last = self.enc_gru(enc_in)

        ctx = h_last.squeeze(0)

        h = ctx

        prev_flow = x_hist[:, -1, :, 0:1].reshape(B*N,1)

        outputs = []

        for k in range(12):

            weather_k = x_future[:, k].reshape(B*N,4)

            if x_future_raw is not None:
                weather_raw = x_future_raw[:, k].reshape(B*N,4)
            else:
                weather_raw = weather_k

            flow_emb = self.flow_proj(prev_flow)
            weather_emb = self.weather_proj(weather_k)

            dec_in = torch.cat([flow_emb, weather_emb, ctx], dim=-1)

            h = self.dec_cell(dec_in, h)

            base_pred = self.out_head(h)

            # ===== 天气分量 =====
            temp = weather_raw[:, 0:1]
            prcp = weather_raw[:, 1:2]
            wdsp = weather_raw[:, 2:3]
            rh = weather_raw[:, 3:4]

            # ===== 多灾种天气影响强度 =====
            rain_stress = F.relu(prcp * 20.0)
            wind_stress = F.relu(wdsp / 5.0)
            cold_stress = F.relu((2.0 - temp) / 5.0)
            snow_ice_stress = cold_stress * F.relu((rh - 85.0) / 15.0)

            bad_weather = (
                    rain_stress +
                    0.5 * wind_stress +
                    0.8 * cold_stress +
                    1.0 * snow_ice_stress
            )

            # ===== gate =====
            gate = self.weather_gate(weather_emb)

            scale = F.softplus(self.weather_scale)

            penalty = self.weather_force * scale * gate * bad_weather

            # ===== 最终输出（防负数）=====
            pred = base_pred - penalty

            outputs.append(pred.view(B,N,1))

            prev_flow = pred

        return torch.stack(outputs, dim=1)