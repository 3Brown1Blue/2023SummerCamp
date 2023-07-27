import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from models.neuconw import SDFNetwork,SingleVarianceNetwork

# xyz and dir embedding
class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        # dim_in:(B,3)
        out = [x]

        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        # dim_out:(B, 6*N_freqs+3)
        return torch.cat(out, -1)

class LightCodeNetwork(nn.Module):
    def __init__(
            self,
            flag,
            encode_shadow=True,
            encode_appearance=False, 
            encode_transient=False, 
            layers=8, 
            hidden=256,
            skips=[4],
            in_channels_xyz=63, 
            in_channels_dir=27,
            in_channels_a=48,
            in_channels_sph=9,
            in_channels_t=16,
            N_emb_xyz=10,
            N_emb_dir=4 
        ):
        super().__init__()
        self.flag=flag
        self.layers=layers
        self.hidden=hidden
        self.skips=skips
        self.in_channels_xyz=in_channels_xyz
        self.in_channels_dir=in_channels_dir
        
        if flag=='fine':
            self.encode_shadow=encode_shadow
            self.encode_appearance=encode_appearance
            self.encode_transient=encode_transient
            self.in_channels_a=in_channels_a
            self.in_channels_t=in_channels_t
            self.in_channels_sph=in_channels_sph
        else:
            self.encode_shadow=False
            self.encode_appearance=False
            self.encode_transient=False
            self.in_channels_a=0
            self.in_channels_t=0
            self.in_channels_sph=0
        
        # points & views embedding
        self.xyz_embedding=PosEmbedding(N_emb_xyz-1,N_emb_xyz)
        self.dir_embedding=PosEmbedding(N_emb_dir-1,N_emb_dir)

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
        
        # direction&appearance-encoder
        dir_a_encoder=[[nn.Linear(hidden+in_channels_dir+self.in_channels_a, hidden//2),nn.ReLU(True)]]+ \
                    [[nn.Linear(hidden//2,hidden//2),nn.ReLU(True)] for _ in range(layers//2)]
        self.dir_a_encoder=nn.Sequential(*sum(dir_a_encoder,[]))

        # shadow layers
        if self.encode_shadow:
            shadow_layers=[]
            dim=in_channels_sph+hidden
            for i in range(1):
                shadow_layers.append(nn.Linear(dim,hidden//2))
                shadow_layers.append(nn.ReLU(True))
                dim=hidden//2
            shadow_layers.append(nn.Linear(dim,1))
            shadow_layers.append(nn.Sigmoid())
            self.shadow_layers=nn.Sequential(*shadow_layers)

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(hidden//2, 3), nn.Sigmoid())

        ## transient(not adding shadow yet)
        if self.encode_transient:
            # transient-encoder
            transient_layers=[]
            dim=hidden+in_channels_t
            for i in range(4):
                transient_layers.append(nn.Linear(dim,hidden//2))
                transient_layers.append(nn.ReLU(True))
                dim=hidden//2
            self.transient_encoder = nn.Sequential(*transient_layers)

            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(hidden//2, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(hidden//2, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(hidden//2, 1), nn.Softplus())

    def forward(self,input_xyz,views,input_a,input_sph,input_t=None,output_transient=False):
        # output the transient code or not
        input_xyz=self.xyz_embedding(input_xyz)
        views=self.dir_embedding(views)
        input_dir_a=torch.cat([views,input_a],dim=-1)
        
        xyz_copy = input_xyz
        for i in range(self.layers):
            if i in self.skips:
                xyz_copy = torch.cat([input_xyz, xyz_copy], -1)
            xyz_copy = getattr(self, f"xyz_encoder_{i+1}")(xyz_copy)

        static_sigma = self.static_sigma(xyz_copy) # B*1

        xyz_encoding_final = self.xyz_encoder_final(xyz_copy)
        rgb_input = torch.cat([xyz_encoding_final, input_dir_a], -1)
        rgb_input = self.dir_a_encoder(rgb_input)
        static_rgb = self.static_rgb(rgb_input) # B*3

        shadow=self.shadow_layers(torch.cat([xyz_encoding_final,input_sph],dim=-1))
        shadow=shadow.repeat((1,)*(len(shadow.size())-1)+(3,))

        if not output_transient:
            if self.encode_shadow:
                return static_sigma,static_rgb,shadow
            else:
                return static_sigma,static_rgb

        ### (not adding shadow yet)
        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], -1)
        transient_encoding = self.transient_encoder(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding) # B*1
        transient_rgb = self.transient_rgb(transient_encoding) # B*3
        transient_beta = self.transient_beta(transient_encoding) # B*1

        return static_sigma,static_rgb,transient_sigma,transient_beta,transient_rgb


'''
    NeRFOSR: recurrented from paper {NeRF for Outdoor Scene Relighting},ECCV 2022  
    url:https://4dqv.mpi-inf.mpg.de/NeRF-OSR/  
    composition:  
    ShadowNet------------------->  
    |                           |  
    NeRFBase(LightCodeNetwork)——DensityNet------>(merge) ---> Output  
    |                           |  
    AlbedoNet------------------->  
'''
class NeRFOSR(nn.Module):
    def __init__(self) :
        super().__init__()

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

        # self.lightcode_net_coarse = LightCodeNetwork(
        #     flag='coarse',
        #     **self.lightcodeNet_config
        # )
        self.lightcode_net_fine = LightCodeNetwork(
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
        input_xyz, view_dirs, input_a, input_sph= torch.split(
            x, [3,3,self.in_channels_a,9], dim=-1
        )
        
        n_rays, n_samples, _ = input_xyz.size()
        input_a = input_a.view(n_rays * n_samples, -1)

        # geometry prediction
        sdf_nn_output = self.sdf_net(input_xyz)  # (B, 1), (B, W)
        static_sdf = sdf_nn_output[:, :1]
        xyz_ = sdf_nn_output[:, 1:]

        # color prediction
        static_gradient = self.gradient(input_xyz)
        static_sigma,static_rgb,shadow  = self.lightcode_net_fine(
            input_xyz.view(-1,3),
            view_dirs.view(-1,3),
            input_a,
            input_sph.view(-1,9)
        )  # (B, 1+3+3)

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
            static_sigma.view(n_rays,n_samples),
            shadow.view(n_rays,n_samples,1)
        )

        return static_out