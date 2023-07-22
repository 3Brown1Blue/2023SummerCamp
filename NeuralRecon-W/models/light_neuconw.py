import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from neuconw import SDFNetwork,SingleVarianceNetwork

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim

class LightCodeNetwork(nn.Module):
    def __init__(
            self,
            flag,
            layers=8, 
            hidden=256, 
            skips=[4],
            in_channels_xyz=63, 
            in_channels_dir=27,
            encode_appearance=False, 
            in_channels_a=48,
            encode_transient=False, 
            in_channels_t=16,
            beta_min=0.03 
        ):
        super().__init__()
        self.flag=flag
        self.layers=layers
        self.hidden=hidden
        self.skips=skips
        self.in_channels_xyz=in_channels_xyz
        self.in_channels_dir=in_channels_dir
        
        if flag=='fine':
            self.encode_appearance=encode_appearance
            self.encode_transient=encode_transient
            self.inchannels_a=in_channels_a
            self.inchannels_t=in_channels_t
        else:
            self.encode_appearance=False
            self.encode_transient=False
            self.inchannels_a=0
            self.inchannels_t=0

        ## static 
        # xyz-encoder
        for i in range(layers):
            if i==0:
                l=nn.Linear(in_channels_xyz,hidden)
            elif i in skips:
                l=nn.Linear(hidden+in_channels_xyz,hidden)
            else:
                l=nn.Linear(hidden,hidden)
            l=nn.Sequential(l,nn.ReLU(inplace=True))
            setattr(self,f"xyz_encoder_{i+1}",l)
        self.xyz_encoder_final=nn.Linear(hidden,hidden)

         # direction-encoder
        self.dir_encoder = nn.Sequential(
                                nn.Linear(hidden+in_channels_dir+self.in_channels_a, hidden//2), 
                                nn.ReLU(True))

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(hidden//2, 3), nn.Sigmoid())

        ## transient
        if self.encode_transient:
            # transient-encoder
            self.transient_encoder = nn.Sequential(
                                        nn.Linear(hidden+in_channels_t, hidden//2), nn.ReLU(True),
                                        nn.Linear(hidden//2, hidden//2), nn.ReLU(True),
                                        nn.Linear(hidden//2, hidden//2), nn.ReLU(True),
                                        nn.Linear(hidden//2, hidden//2), nn.ReLU(True))
            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(hidden//2, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(hidden//2, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(hidden//2, 1), nn.Softplus())

    def forward(self, x,output_transient=True):
        # output the transient code or not
        if output_transient:
            input_xyz, input_dir_a, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a,
                                self.in_channels_t], dim=-1)
        else:
            input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a], dim=-1)

        xyz_copy = input_xyz
        for i in range(self.layers):
            if i in self.skips:
                xyz_copy = torch.cat([input_xyz, xyz_copy], 1)
            xyz_copy = getattr(self, f"xyz_encoder_{i+1}")(xyz_copy)

        static_sigma = self.static_sigma(xyz_copy) # B*1

        xyz_encoding_final = self.xyz_encoder_final(xyz_copy)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoder(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding) # B*3
        static = torch.cat([static_rgb, static_sigma], 1)

        if not output_transient:
            return static

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoder(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding) # B*1
        transient_rgb = self.transient_rgb(transient_encoding) # B*3
        transient_beta = self.transient_beta(transient_encoding) # B*1

        transient = torch.cat([transient_rgb, transient_sigma,
                               transient_beta], 1) 

        return torch.cat([static, transient], 1) 
      

class LightNeuconW(nn.Module):
    def __init__(
        self,
        sdfNet_config,
        lightcodeNet_config,
        SNet_config,
        in_channels_a,
        encode_a,
    ):

        super().__init__()
        self.sdfNet_config = sdfNet_config
        self.lightcodeNet_config = lightcodeNet_config
        self.SNet_config = SNet_config
        self.in_channels_a = in_channels_a
        self.encode_a = encode_a

        # xyz encoding layers + sdf layer
        self.sdf_net = SDFNetwork(**self.sdfNet_config)

        self.xyz_encoding_final = nn.Linear(512, 512)

        # Static deviation
        self.deviation_network = SingleVarianceNetwork(**self.SNet_config)

        # (static & transient) Light code and color based on Nerf-W
        # optimize 2 models at same time
        self.lightcode_net_coarse = LightCodeNetwork(
            flag='coarse',
            **self.lightcodeNet_config
        )
        self.lightcode_net_fine = LightCodeNetwork(
            flag='fine',
            **self.lightcodeNet_config
        )

    def sdf(self, input_xyz):
        # geometry prediction
        return self.sdf_net.sdf(input_xyz)  # (B, w+1)
        # return static_sdf[:, 1], static_sdf[:, 1:]

    def gradient(self, x):
        return self.sdf_net.gradient(x)

    def forward(self, x):
        device = x.device
        input_xyz, view_dirs, input_dir_a = torch.split(
            x, [3, 3, self.in_channels_a], dim=-1
        )

        n_rays, n_samples, _ = input_xyz.size()
        input_dir_a = input_dir_a.view(n_rays * n_samples, -1)

        # geometry prediction
        sdf_nn_output = self.sdf_net(input_xyz)  # (B, 1), (B, W)
        static_sdf = sdf_nn_output[:, :1]
        xyz_ = sdf_nn_output[:, 1:]

        # color prediction
        static_gradient = self.gradient(input_xyz)
        static_rgb, xyz_encoding_final, view_encoded = self.lightcode_net_net(
            input_xyz.view(-1, 3),
            static_gradient.view(-1, 3),
            view_dirs.view(-1, 3),
            xyz_,
            input_dir_a,
        )  # (B, 3)
        # sdf gradient
        static_deviation = self.deviation_network(torch.zeros([1, 3], device=device))[
            :, :1
        ].clamp(
            1e-6, 1e6
        )  # (B, 1)

        static_out = (
            static_rgb.view(n_rays, n_samples, 3),
            static_deviation,
            static_sdf.view(n_rays, n_samples),
            static_gradient.view(n_rays, n_samples, 3),
        )

        return static_out