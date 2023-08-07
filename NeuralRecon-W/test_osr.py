import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import metrics
import yaml

from tqdm import tqdm
from collections import defaultdict
from argparse import ArgumentParser

from models.nerf import NeRF
from models.light_neuconw import LightNeuconW
from rendering.custom_renderer import LightNeuconWRenderer

from datasets import dataset_dict
from utils import load_ckpt
from utils.comm import *
from utils.visualization import extract_mesh

from config.defaults import get_cfg_defaults

def get_opts():
    parser=ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='config/train_brandenburg_gate.yaml',
                        help='config path')
    parser.add_argument('--ckpt_path',type=str,default='ckpts/train-osr_728-20230728_134619/{epoch:d}/epoch=0-step=409999.ckpt',
                        help='load a checkpoint from $CKPT_PATH')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--num_workers',type=int,default=4,
                        help='number of workers')
    parser.add_argument('--chunks', type=int, default=32,
                        help='chunk size(avoid OOM)')
    parser.add_argument('--N_vocab',type=int,default=5000,
                        help='vocab size')
    parser.add_argument('encode_transient',default=False,action='store_true',
                        help='whether add transient embedding')
    parser.add_argument('--N_t',type=int,default=16,
                        help='apperance embbeding size')
    parser.add_argument('--N_a',type=int,default=48,
                        help='apperance embbeding size')

    return parser.parse_args()

def main(hparams,config):
    # config of renderer
    scene_config_path = os.path.join(config.DATASET.ROOT_DIR, "config.yaml")
    with open(scene_config_path, "r") as yamlfile:
        scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    spc_options = {
        "voxel_size": scene_config["voxel_size"],
        "recontruct_path": config.DATASET.ROOT_DIR,
        "min_track_length": scene_config["min_track_length"],
    }

    # appearance embeddings
    embeddings={}
    embedding_a=torch.nn.Embedding(hparams.N_vocab,hparams.N_a).cuda()
    embeddings['a']=embedding_a

    # model
    lightneuconw = LightNeuconW(
        sdfNet_config=config.NEUCONW.SDF_CONFIG,
        lightcodeNet_config=config.LIGHTNEUCONW.LIGHT_CONFIG,
        SNet_config=config.NEUCONW.S_CONFIG,
        in_channels_a=config.NEUCONW.N_A,
        encode_a=config.NEUCONW.ENCODE_A,
    ).cuda()
    nerf = NeRF(
        D=8,
        d_in=4,
        d_in_view=3,
        W=256,
        multires=10,
        multires_view=4,
        output_ch=4,
        skips=[4],
        encode_appearance=config.NEUCONW.ENCODE_A_BG,
        in_channels_a=config.NEUCONW.N_A,
        in_channels_dir=6*config.NEUCONW.COLOR_CONFIG.multires_view+3,
        use_viewdirs=True,
    ).cuda()

    load_ckpt(embedding_a,hparams.ckpt_path,model_name='embedding_a')
    load_ckpt(lightneuconw,hparams.ckpt_path,model_name='lightneuconw')
    load_ckpt(nerf,hparams.ckpt_path,model_name='nerf')

    renderer=LightNeuconWRenderer(
        nerf=nerf,
        lightneuconw=lightneuconw,
        embeddings=embeddings,
        n_samples=config.NEUCONW.N_SAMPLES,
        s_val_base=config.NEUCONW.S_VAL_BASE,
        n_importance=config.NEUCONW.N_IMPORTANCE,
        n_outside=config.NEUCONW.N_OUTSIDE,
        up_sample_steps=config.NEUCONW.UP_SAMPLE_STEP,
        perturb=1.0,
        origin=scene_config["origin"],
        radius=scene_config["radius"],
        render_bg=config.NEUCONW.RENDER_BG,
        mesh_mask_list=config.NEUCONW.MESH_MASK_LIST,
        floor_normal=config.NEUCONW.FLOOR_NORMAL,
        floor_labels=config.NEUCONW.FLOOR_LABELS,
        depth_loss=config.NEUCONW.DEPTH_LOSS,
        spc_options=spc_options,
        sample_range=config.NEUCONW.SAMPLE_RANGE,
        boundary_samples=config.NEUCONW.BOUNDARY_SAMPLES,
        nerf_far_override=config.NEUCONW.NEAR_FAR_OVERRIDE,
    )

    # get rendered image
    def inference(rays,ts,label,a_embedded):
        results=defaultdict(list)
        B=rays.shape[0]
        chunks=hparams.chunks

        # split input into chunks to avoid CUDA OOM
        for i in tqdm(range(0,B,chunks)):
            rendered_ray_chunks=renderer.render(
                rays[i:i+chunks],
                ts[i:i+chunks],
                label[i:i+chunks],
                background_rgb=torch.zeros([1, 3], device=rays.device),
                a_embedded=a_embedded[i:i+chunks]
            )
            for k,v in rendered_ray_chunks.items():
                results[k]+=[v]
        
        for k,v in results.items():
            results[k]=torch.cat(v,dim=0)

        return results
    
    dataset=dataset_dict[config.DATASET.DATASET_NAME]\
            (
                root_dir=config.DATASET.ROOT_DIR,
                semantic_map_path=config.DATASET.PHOTOTOURISM.SEMANTIC_MAP_PATH,
                with_semantics=config.DATASET.PHOTOTOURISM.WITH_SEMANTICS,
                split='test_train',
                img_downscale=8,
                use_cache=False
            )
    
    sample1=dataset[62]

    # only load rays of sample1 to save Memory
    rays1=sample1['rays'].cuda()
    ts1=sample1['ts'].cuda()
    label1=sample1['semantics'].cuda()
    a_embedded1=embedding_a(ts1)

    results=inference(rays1, ts1, label1, a_embedded1)
    
    img_wh=tuple(sample1['img_wh'].numpy())
    albedo_map=results['fg_albedo'].view(img_wh[1],img_wh[0],3).cpu().detach().numpy()
    shadow_map=results['fg_shadow'].view(img_wh[1],img_wh[0],3).cpu().detach().numpy()
    normal_map=results['fg_normal'].view(img_wh[1],img_wh[0],3).cpu().detach().numpy()
    rgb_map=results['fg_rgb'].view(img_wh[1],img_wh[0],3).cpu().detach().numpy()

    fig,axes=plt.subplots(2,2,figsize=(20, 20),tight_layout=True)

    axes[0,0].imshow(rgb_map)
    axes[0,0].axis('off')
    axes[0,0].set_title('rgb_map')

    axes[0,1].imshow(normal_map)
    axes[0,1].axis('off')
    axes[0,1].set_title('normal_map')

    axes[1,0].imshow(albedo_map)
    axes[1,0].axis('off')
    axes[1,0].set_title('albedo_map')

    axes[1,1].imshow(shadow_map)
    axes[1,1].axis('off')
    axes[1,1].set_title('shadow_map')

    fig.savefig('osr.jpg')



if __name__ == '__main__':
    hparams = get_opts()

    config = get_cfg_defaults()
    config.merge_from_file(hparams.cfg_path)

    main(hparams,config)