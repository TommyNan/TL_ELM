# -*- coding: utf-8 -*-
"""NeRF2 NN model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.wesn import WESN
# from einops import rearrange

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2me = lambda x, y : torch.mean(abs(x - y))
sig2mse = lambda x, y : torch.mean((x - y) ** 2)
csi2snr = lambda x, y: -10 * torch.log10(
    torch.norm(x - y, dim=(1, 2)) ** 2 /
    torch.norm(y, dim=(1, 2)) ** 2
)


class Embedder():
    """positional encoding"""
    def __init__(self, input_dims, num_freq, include_input=True, log_sampling=True) -> None:
        self.input_dims = input_dims
        self.num_freq = num_freq
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.periodic_fns = [torch.sin, torch.cos]
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        out_dim = 0

        if self.include_input:
            embed_fns.append(lambda x : x)
            out_dim += self.input_dims

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., self.num_freq-1, steps=self.num_freq)  #2^[0,1,...,L-1]
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.num_freq-1), steps=self.num_freq)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """return: gamma(input)
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(num_freq, if_embeded=True, input_dims=3):
    """get positional encoding function

    Parameters
    ----------
    num_freq : log2 of max freq for positional encoding, i.e., (L-1)
    if_embeded : set 1 for default positional encoding
    input_dims : input dimension of gamma

    Returns
    -------
        embedding function; output_dims input_dim*(2L+1)
    """
    if if_embeded == False:
        return nn.Identity(), input_dims

    embedder_obj = Embedder(input_dims=input_dims, num_freq=num_freq)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim



class NeRF2(nn.Module):

    def __init__(self, model_params):
        """NeRF2 model

        Parameters
        ----------
        num_layers : int, hidden layer number, default by 8
        num_hidden : int, Dimension per hidden layer, default by 256
        skip : list, skip layer index
        input_dims: dict, input dimensions
        num_freq: dict, log2 of max freq for position, view, and tx position positional encoding, i.e., (L-1)
        if_embeded : dict, whether to use positional encoding
        attn_output_dims : int, output dimension of attenuation
        sig_output_dims : int, output dimension of signal
        """

        super().__init__()
        self.num_layers = model_params["num_layers"]
        self.num_hidden = model_params["num_hidden"]
        self.skips = model_params["skips"]
        self.input_dims = model_params["input_dims"]
        self.num_freq = model_params["num_freq"]
        self.if_embeded = model_params["if_embeded"]
        self.attn_output_dims = model_params["attn_output_dims"]
        self.sig_output_dims = model_params["sig_output_dims"]


        # set positional encoding function
        self.pos_embed_pts, self.input_pts_dim = get_embedder(self.num_freq['pts'], self.if_embeded['pts'], self.input_dims['pts'])
        self.pos_embed_view, self.input_view_dim = get_embedder(self.num_freq['view'], self.if_embeded['view'], self.input_dims['view'])
        self.pos_embed_tx, self.input_tx_dim = get_embedder(self.num_freq['tx'], self.if_embeded['tx'], self.input_dims['tx'])

        ## attenuation network
        self.attenuation_linears = nn.ModuleList(
            [nn.Linear(self.input_pts_dim, self.num_hidden)] +
            [nn.Linear(self.num_hidden, self.num_hidden) if i not in self.skips else nn.Linear(self.num_hidden + self.input_pts_dim, self.num_hidden) for i in range(self.num_layers - 1)]
        )

        ## radiance net
        self.radiance_net = nn.ModuleList(
            [nn.Linear(self.input_view_dim + self.input_tx_dim + self.num_hidden, self.num_hidden)] +
            [nn.Linear(self.num_hidden, self.num_hidden//2)]
        )

        ## output head, 2 for amplitude and phase
        self.attenuation_output = nn.Linear(self.num_hidden, self.attn_output_dims)
        self.feature_layer = nn.Linear(self.num_hidden, self.num_hidden)
        self.signal_output = nn.Linear(self.num_hidden//2, self.sig_output_dims)


    def forward(self, pts, view, tx):
        """forward function of the model

        Parameters
        ----------
        pts: [batch_size, n_samples, 3], position of voxels
        view: [batch_size, n_samples, 3], view direction
        tx: [batch_size, n_samples, 3], position of transmitter

        Returns
        ----------
        outputs: [batch_size, n_samples, 4].   attn_amp, attn_phase, signal_amp, signal_phase
        """

        # position encoding
        pts = self.pos_embed_pts(pts).contiguous()   # [B, N, input_dim*(2L+1)]
        view = self.pos_embed_view(view).contiguous()
        tx = self.pos_embed_tx(tx).contiguous()
        shape = pts.shape
        pts = pts.view(-1, list(pts.shape)[-1])
        view = view.view(-1, list(view.shape)[-1])
        tx = tx.view(-1, list(tx.shape)[-1])

        x = pts
        for i, layer in enumerate(self.attenuation_linears):
            x = F.relu(layer(x))
            if i in self.skips:
                x = torch.cat([pts, x], -1)

        attn = self.attenuation_output(x)    # (batch_size, num_samples,  2)
        feature = self.feature_layer(x)
        x = torch.cat([feature, view, tx], -1)

        for i, layer in enumerate(self.radiance_net):
            x = F.relu(layer(x))
        signal = self.signal_output(x)    #[batch_size * n_samples, 2]

        outputs = torch.cat([attn, signal], -1).contiguous()    # [batch_size * num_points, 4]
        return outputs.view(shape[:-1]+outputs.shape[-1:])

    def summary(self):
        lines = []

        lines.append("### *** Model Overview\n")
        lines.append(f"Inputs:")
        lines.append(f"  - pts:  [B, {self.input_pts_dim}] (after embedding)")
        lines.append(f"  - view: [B, {self.input_view_dim}] (after embedding)")
        lines.append(f"  - tx:   [B, {self.input_tx_dim}] (after embedding)\n")

        lines.append("Positional Embedding:")
        lines.append(f"  - pts_embed   : [B, {self.input_pts_dim}]")
        lines.append(f"  - view_embed  : [B, {self.input_view_dim}]")
        lines.append(f"  - tx_embed    : [B, {self.input_tx_dim}]\n")

        lines.append("### *** Attenuation Network (attenuation_linears)\n")
        lines.append(f"  num_layers: {self.num_layers}")
        lines.append(f"  num_hidden: {self.num_hidden}")
        lines.append(f"  skips: {self.skips}\n")

        lines.append(f"  Layer 0: Linear({self.input_pts_dim} -> {self.num_hidden})")
        for i in range(1, self.num_layers):
            if i in self.skips:
                lines.append(
                    f"  Layer {i}: Linear({self.num_hidden} + {self.input_pts_dim} -> {self.num_hidden})  # skip connection"
                )
            else:
                lines.append(f"  Layer {i}: Linear({self.num_hidden} -> {self.num_hidden})")
        lines.append(f"\n  attenuation_output: Linear({self.num_hidden} -> {self.attn_output_dims})")
        lines.append(f"  feature_layer     : Linear({self.num_hidden} -> {self.num_hidden})\n")

        lines.append("### *** Radiance Network (radiance_net)\n")
        rad_input_dim = self.input_view_dim + self.input_tx_dim + self.num_hidden
        lines.append(f"  Input dim: {rad_input_dim} = view({self.input_view_dim}) + tx({self.input_tx_dim}) + feat({self.num_hidden})")
        lines.append(f"  Layer 0: Linear({rad_input_dim} -> {self.num_hidden})")
        lines.append(f"  Layer 1: Linear({self.num_hidden} -> {self.num_hidden // 2})")
        lines.append(f"\n  signal_output: Linear({self.num_hidden // 2} -> {self.sig_output_dims})")
        return "\n".join(lines)



class NeRF2_ESN_V1(nn.Module):

    def __init__(self, model_params):
        """NeRF2 model

        Parameters
        ----------
        num_layers : int, hidden layer number, default by 8
        num_hidden : int, Dimension per hidden layer, default by 256
        skip : list, skip layer index
        input_dims: dict, input dimensions
        num_freq: dict, log2 of max freq for position, view, and tx position positional encoding, i.e., (L-1)
        if_embeded : dict, whether to use positional encoding
        attn_output_dims : int, output dimension of attenuation
        sig_output_dims : int, output dimension of signal
        """

        super().__init__()
        self.num_layers = model_params["num_layers"]
        self.num_hidden = model_params["num_hidden"]

        self.num_hidden_wesn= model_params["num_hidden_wesn"]
        self.connectivity = model_params["connectivity"]
        self.leaky = model_params["leaky"]
        self.spectral_radius = model_params["spectral_radius"]
        self.use_norm2 = model_params["use_norm2"]
        self.use_bias = model_params["use_bias"]
        self.win_len = model_params["win_len"]

        self.skips = model_params["skips"]
        self.input_dims = model_params["input_dims"]
        self.num_freq = model_params["num_freq"]
        self.if_embeded = model_params["if_embeded"]
        self.attn_output_dims = model_params["attn_output_dims"]
        self.sig_output_dims = model_params["sig_output_dims"]
        self.feature_dim = self.num_hidden
        wesn_out_dim = self.attn_output_dims + self.feature_dim

        # set positional encoding function
        self.pos_embed_pts, self.input_pts_dim = get_embedder(self.num_freq['pts'], self.if_embeded['pts'], self.input_dims['pts'])
        self.pos_embed_view, self.input_view_dim = get_embedder(self.num_freq['view'], self.if_embeded['view'], self.input_dims['view'])
        self.pos_embed_tx, self.input_tx_dim = get_embedder(self.num_freq['tx'], self.if_embeded['tx'], self.input_dims['tx'])

        self.attenuation_wesn = WESN(
            hidden_dim=self.num_hidden_wesn,
            connectivity=self.connectivity,
            leaky=self.leaky,
            spectral_radius=self.spectral_radius,
            use_norm2=self.use_norm2,
            use_bias=self.use_bias,
            win_len=self.win_len,
            output_dim=wesn_out_dim,
        )

        ## radiance net
        self.radiance_net = nn.ModuleList(
            [nn.Linear(self.input_view_dim + self.input_tx_dim + self.feature_dim, self.num_hidden)] +
            [nn.Linear(self.num_hidden, self.num_hidden//2)]
        )

        ## output head, 2 for amplitude and phase
        self.signal_output = nn.Linear(self.num_hidden//2, self.sig_output_dims)


    def forward(self, pts, view, tx):
        """forward function of the model

        Parameters
        ----------
        pts: [batch_size, n_samples, 3], position of voxels
        view: [batch_size, n_samples, 3], view direction
        tx: [batch_size, n_samples, 3], position of transmitter

        Returns
        ----------
        outputs: [batch_size, n_samples, 4].   attn_amp, attn_phase, signal_amp, signal_phase
        """

        if pts.dim() == 3:
            # [B, N, 3] -> [B, 1, N, 3]
            pts = pts.unsqueeze(1)
            view = view.unsqueeze(1)
            tx = tx.unsqueeze(1)
            squeeze_rays = True
        elif pts.dim() == 4:
            # [B, R, N, 3]
            squeeze_rays = False
        else:
            raise ValueError(f"Expected pts/view/tx to be 3D or 4D, got pts.dim()={pts.dim()}")

        B, R, N, _ = pts.shape

        # position encoding
        pts_flat = pts.reshape(B * R, N, 3)
        view_flat = view.reshape(B * R, N, 3)
        tx_flat = tx.reshape(B * R, N, 3)

        pts_e = self.pos_embed_pts(pts_flat).contiguous()  # [B*R, N, Dp]
        view_e = self.pos_embed_view(view_flat).contiguous()  # [B*R, N, Dv]
        tx_e = self.pos_embed_tx(tx_flat).contiguous()  # [B*R, N, Dtx]


        BR, N2, Dp = pts_e.shape
        _, _, Dv = view_e.shape
        _, _, Dtx = tx_e.shape
        assert BR == B * R and N2 == N, "Unexpected shape after flattening/embedding."

        # ------------------------------------------------------------
        # 1) Attenuation + Feature via WESN (sequence model)
        #    WESN input:  [B, N, Dp]
        #    WESN output: [B, N, attn_out + feature_dim]
        # ------------------------------------------------------------
        wesn_out = self.attenuation_wesn(pts_e)
        attn = wesn_out[..., :self.attn_output_dims]
        feature = wesn_out[..., self.attn_output_dims:]

        # ------------------------------------------------------------
        # 2) Radiance network (pointwise MLP): use feature + view + tx
        #    flatten to [B*N, ...] as before
        # ------------------------------------------------------------
        feature_flat = feature.reshape(B * R * N, self.feature_dim)  # [B*R*N, feature_dim]
        view_feat = view_e.reshape(B * R * N, Dv)  # [B*R*N, Dv]
        tx_feat = tx_e.reshape(B * R * N, Dtx)  # [B*R*N, Dtx]

        x = torch.cat([feature_flat, view_feat, tx_feat], dim=-1)  # [B*R*N, feature_dim + Dv + Dtx]

        for layer in self.radiance_net:
            x = F.relu(layer(x))

        signal = self.signal_output(x)  # [B*N, sig_output_dims]
        signal = signal.reshape(B * R, N, self.sig_output_dims)  # [B*R, N, sig_out]

        outputs = torch.cat([attn, signal], dim=-1).contiguous()  # [B*R, N, attn_out + sig_out]

        C = outputs.shape[-1]
        outputs = outputs.reshape(B, R, N, C)

        if squeeze_rays:
            outputs = outputs.squeeze(1)  # [B, 1, N, C] -> [B, N, C]

        return outputs

    def summary(self):
        lines = []

        lines.append("### *** Model Overview\n")
        lines.append(f"Inputs:")
        lines.append(f"  - pts:  [B, {self.input_pts_dim}] (after embedding)")
        lines.append(f"  - view: [B, {self.input_view_dim}] (after embedding)")
        lines.append(f"  - tx:   [B, {self.input_tx_dim}] (after embedding)\n")

        lines.append("Positional Embedding:")
        lines.append(f"  - pts_embed   : [B, {self.input_pts_dim}]")
        lines.append(f"  - view_embed  : [B, {self.input_view_dim}]")
        lines.append(f"  - tx_embed    : [B, {self.input_tx_dim}]\n")

        lines.append("### *** Attenuation Network (attenuation_ESN)\n")
        lines.append(f"  num_reservoir: {self.num_hidden_wesn}")
        lines.append(f"  connectivity: {self.connectivity}")
        lines.append(f"  leaky: {self.leaky}")
        lines.append(f"  spectral_radius: {self.spectral_radius}")
        lines.append(f"  use_norm2: {self.use_norm2}")
        lines.append(f"  use_bias: {self.use_bias}")
        lines.append(f"  win_len: {self.win_len}\n")

        lines.append("### *** Radiance Network (radiance_net)\n")
        rad_input_dim = self.input_view_dim + self.input_tx_dim + self.num_hidden
        lines.append(f"  Input dim: {rad_input_dim} = view({self.input_view_dim}) + tx({self.input_tx_dim}) + feat({self.num_hidden})")
        lines.append(f"  Layer 0: Linear({rad_input_dim} -> {self.num_hidden})")
        lines.append(f"  Layer 1: Linear({self.num_hidden} -> {self.num_hidden // 2})")
        lines.append(f"\n  signal_output: Linear({self.num_hidden // 2} -> {self.sig_output_dims})")
        return "\n".join(lines)