import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import tinycudann as tcnn


class VanillaMLP(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 n_neurons,
                 n_hidden_layers,
                 sphere_init=False,
                 weight_norm=False,
                 sphere_init_radius=0.5,
                 output_activation='identity'):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = n_neurons, n_hidden_layers
        self.sphere_init, self.weight_norm = sphere_init, weight_norm
        self.sphere_init_radius = sphere_init_radius
        self.layers = [self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False), self.make_activation()]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False),
                            self.make_activation()]
        self.layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x.float())
        return -x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True)  # network without bias will degrade quality
        if self.sphere_init:
            if is_last:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(dim_in), std=0.0001)
            elif is_first:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(dim_out))
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(dim_out))
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)



class SpherelatentField(nn.Module):
    def __init__(self,
                 n_levels=16,
                 log2_hashmap_size=19,
                 base_res=16,
                 fine_res=192,
                 channel_n=14*4):
        super().__init__()
        per_level_scale = np.exp(np.log(fine_res / base_res) / (n_levels - 1))
        self.hash_grid = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": per_level_scale,
                "interpolation": "Smoothstep",
            }
        )

        self.latent_mlp = VanillaMLP(dim_in=3, #n_levels * 2 + 3,
                                  dim_out=channel_n,
                                  n_neurons=64,
                                  n_hidden_layers=3,
                                  sphere_init=True,
                                  weight_norm=False)

    def forward(self, directions, requires_grad=False):
        latents = self.latent_mlp(directions)
        return latents

        if requires_grad:
            if not self.training:
                directions = directions.clone()  # get a copy to enable grad
            directions.requires_grad_(True)

        dir_scaled = directions * 0.49 + 0.49
        selector = ((dir_scaled > 0.0) & (dir_scaled < 1.0)).all(dim=-1).to(torch.float32)
        scene_feat = self.hash_grid(dir_scaled)

        # distance = F.softplus(self.geo_mlp(torch.cat([directions, scene_feat], -1))[..., 0] + 1.)
        # distance = torch.exp(self.geo_mlp(torch.cat([directions, scene_feat], -1))[..., 0])
        latents = self.latent_mlp(torch.cat([directions, scene_feat], -1))#[..., 0] + 1.
        if requires_grad:
            grad = torch.autograd.grad(
                latents, directions, grad_outputs=torch.ones_like(latents),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]

            return latents, grad
        else:
            return latents