import torch
from torch import nn
from pt_tnn.temporal_graph import TemporalGraph
from torch.nn.modules.transformer import MultiheadAttention
import einops


class TNNEncoder(nn.Module):
    def __init__(self, n_times):
        super(TNNEncoder, self).__init__()
        self.TG = TemporalGraph(
            model_config_file="configs/test_resnet18_bn_timevary.json",
            input_shape=[3, 224, 224],
            num_timesteps=n_times,
        )
        self.n_time = n_times

    def forward(self, x):
        return self.TG(x, n_times=self.n_time, return_all=True)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        attn_output, _ = self.attn(
            query=x, key=x, value=x
        )  # self-attention: query=key=value=x
        out = self.norm(x + attn_output)  # Add & Norm
        return out


class LazyLinearDecoder(nn.Module):
    def __init__(self, out_features):
        super(LazyLinearDecoder, self).__init__()
        self.lin = nn.LazyLinear(out_features=out_features)

    def forward(self, x):
        x = einops.rearrange(x, "b t d -> b (t d)")
        return self.lin(x)


class EncAttDec(nn.Module):
    def __init__(self, out_features, n_times=10):
        super(EncAttDec, self).__init__()

        self.encoder = TNNEncoder(n_times=n_times)
        # maps (bs, t, C, H, W) ---> (bs, t, d) or (bs, C, H, W) ---> (bs, d)
        self.attender = SelfAttention(
            embed_dim=1000, num_heads=8  # the output dimension of ResNet18 is 1000
        )
        # maps (bs, t, d) ---> (bs, t, d) or identity [e.g., (bs, d) ---> (bs, d)]
        self.decoder = LazyLinearDecoder(out_features=out_features)
        # maps (bs, t, d) ---> (bs, num_classes) or identity [e.g., (bs, d) ---> (bs, d)]

    def forward(
        self,
        x,
    ):
        # encode, (bs, t, C, H, W) ---> (bs, t, d) or (bs, C, H, W) ---> (bs, d)
        x = self.encoder(x)
        # attend, (bs, t, d) ---> (bs, t, d) or identity [e.g., (bs, d) ---> (bs, d)]
        x = self.attender(x)
        # decode/predict, (bs, t, d) ---> (bs, num_classes) or identity [e.g., (bs, d) ---> (bs, d)]
        pred = self.decoder(x)

        return pred

if __name__ == "__main__":
    bs, T = 3, 10
    random_input = torch.rand(bs, T, 3, 224, 224)
    out_features = 100
    
    
    model = EncAttDec(
        out_features=out_features,  # output dimension
        n_times=T,  # number of unrolling steps in the TNN encoder
    )
    output = model(random_input)
    
    print(output.shape)  # (bs, out_features)
