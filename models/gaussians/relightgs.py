import os
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.general_utils import rotation_to_quaternion, quaternion_multiply
from utils.sh_utils import RGB2SH, eval_sh
from utils.system_utils import mkdir_p
from tqdm import tqdm
from omegaconf import OmegaConf
from typing import Dict, List, Tuple
import logging
from utils.general_utils import strip_symmetric, build_scaling_rotation
from models.gaussians.basics import *
logger = logging.getLogger()
# from bvh import RayTracer
from utils.graphics_utils import sample_incident_rays
from bvh import RayTracer
import open3d as o3d



def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

class RelightableGaussian(nn.Module):

    def __init__(
        self,
        class_name: str,
        ctrl: OmegaConf,
        reg: OmegaConf = None,
        networks: OmegaConf = None,
        scene_scale: float = 30.,
        scene_origin: torch.Tensor = torch.zeros(3),
        num_train_images: int = 300,
        device: torch.device = torch.device("cuda"),
        **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.ctrl_cfg = ctrl
        self.reg_cfg = reg
        self.networks_cfg = networks
        self.scene_scale = scene_scale
        self.scene_origin = scene_origin
        self.num_train_images = num_train_images
        self.step = 0
        self.raytracer = None
        self.device = device
        self.ball_gaussians=self.ctrl_cfg.get("ball_gaussians", False)
        self.gaussian_2d = self.ctrl_cfg.get("gaussian_2d", False)
        
        # for evaluation
        self.in_test_set = False
        
        # init models
        self.xys_grad_norm = None
        self.max_2Dsize = None
        self._means = torch.zeros(1, 3, device=self.device)
        if self.ball_gaussians:
            self._scales = torch.zeros(1, 1, device=self.device)
        else:
            if self.gaussian_2d:
                self._scales = torch.zeros(1, 2, device=self.device)
            else:
                self._scales = torch.zeros(1, 3, device=self.device)
        self._quats = torch.zeros(1, 4, device=self.device)
        self._opacities = torch.zeros(1, 1, device=self.device)
        self._features_dc = torch.zeros(1, 3, device=self.device)
        self._features_rest = torch.zeros(1, num_sh_bases(self.sh_degree) - 1, 3, device=self.device)
        
        self.use_pbr = True
        if self.use_pbr:
            self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)
            self.base_color_activation = lambda x: torch.sigmoid(x) 
            self.roughness_activation = lambda x: torch.sigmoid(x)
            self.reflectivity_activation = lambda x: torch.sigmoid(x)
            self.sun_visibility_activation = lambda x: torch.sigmoid(x)

            self._normals = torch.zeros(1, 3, device=self.device)
            self._base_color = torch.zeros(1, 4, device=self.device)
            self._roughness = torch.zeros(1, 1, device=self.device)
            #self._metallic = torch.zeros(1, 1, device=self.device)
            self.covariance_activation = build_covariance_from_scaling_rotation
            self._reflectivity = torch.zeros(1, 1, device=self.device) 
            self._sun_visibility = torch.zeros(1, 1, device=self.device)
            self._visibility_tracing = torch.empty(0)
            self._incident_dirs = torch.empty(0)
            self._incident_areas = torch.empty(0)
            self._incidents_dc = torch.zeros(1, 1, 3, device=self.device) #torch.empty(0)
            self._incidents_rest = torch.zeros(1, 15, 3, device=self.device) #torch.empty(0)
            # self._visibility_dc = torch.empty(0)
            # self._visibility_rest = torch.empty(0)
    @property
    def sh_degree(self):
        return self.ctrl_cfg.sh_degree

    def create_from_pcd(self, init_means: torch.Tensor, init_colors: torch.Tensor) -> None:
        self._means = Parameter(init_means)
        
        distances, _ = k_nearest_sklearn(self._means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True).to(self.device)
        if self.ball_gaussians:
            self._scales = Parameter(torch.log(avg_dist.repeat(1, 1)))
        else:
            if self.gaussian_2d:
                self._scales = Parameter(torch.log(avg_dist.repeat(1, 2)))
            else:
                self._scales = Parameter(torch.log(avg_dist.repeat(1, 3)))
        self._quats = Parameter(random_quat_tensor(self.num_points).to(self.device))
        dim_sh = num_sh_bases(self.sh_degree)

        fused_color = RGB2SH(init_colors) # float range [0, 1] 
        shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().to(self.device)
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(init_colors, eps=1e-10)
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        self._opacities = Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1, device=self.device)))

        if self.use_pbr:
            #TODO: calculate normal from pc first
            self._reflectivity = Parameter(torch.zeros(init_means.shape[0], 1, device=self.device))

            self._sun_visibility = Parameter(torch.zeros(init_means.shape[0], 1, device=self.device))

            # Load point cloud
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(init_means.detach().cpu().numpy())
            # Estimate normals
            point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

            # Optional: Orient normals consistently
            point_cloud.orient_normals_consistent_tangent_plane(k=50)
            normal = np.asarray(point_cloud.normals).astype(np.float32)
            sign = (normal[...,2].mean() > 0)
            self._normals = Parameter(torch.tensor(sign * normal, device=self.device))
            #self._normals = Parameter(torch.zeros(init_means.shape[0], 3, device=self.device))
            reflectance = init_colors.mean(dim=-1)[...,None]
            init_colors = torch.cat((init_colors,reflectance),dim=-1)
            self._base_color = Parameter(init_colors)
            #TODO: use lidar intensity to initalize roughness
            self._roughness = Parameter(torch.ones(init_means.shape[0], 1, device=self.device)) 
            #self._metallic = Parameter(torch.zeros(init_means.shape[0], 1, device=self.device))    
            self.max_sh_degree = 3
            incidents = torch.zeros((init_means.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            self._incidents_dc = Parameter(
                incidents[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)) #N*1*3
            self._incidents_rest = Parameter(
                incidents[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)) #N*15*3

            #self.update_visibility()        
        
    @property
    def colors(self):
        if self.sh_degree > 0:
            return SH2RGB(self._features_dc)
        else:
            return torch.sigmoid(self._features_dc)
    @property
    def shs_0(self):
        return self._features_dc
    @property
    def shs_rest(self):
        return self._features_rest
    @property
    def num_points(self):
        return self._means.shape[0]
    @property
    def get_scaling(self):
        if self.ball_gaussians:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales).repeat(1, 2)
                scaling = torch.cat([scaling, torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales).repeat(1, 3)
        else:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales)
                scaling = torch.cat([scaling[..., :2], torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales)

    @property
    def get_xyz(self):
        return self._means

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacities)

    @property
    def get_quats(self):
        return self.quat_act(self._quats)

    @property
    def get_base_color(self):
        return self.base_color_activation(self._base_color[...,:3]) #* self.base_color_scale[None, :]

    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness)

    @property
    def get_reflectivity(self):
        #return self.reflectivity_activation(self._base_color[...,-1][...,None])
        return self.reflectivity_activation(self._reflectivity)

    @property
    def get_sun_visibility(self):
        return self.sun_visibility_activation(self._sun_visibility)


    @property
    def get_incidents(self):
        """SH"""
        incidents_dc = self._incidents_dc
        incidents_rest = self._incidents_rest
        return torch.cat((incidents_dc, incidents_rest), dim=1)

    @property
    def get_normal(self):
        return self.normal_activation(self._normals)


    def quat_act(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True)
    
    def preprocess_per_train_step(self, step: int):
        self.step = step
        
    def postprocess_per_train_step(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        self.after_train(radii, xys_grad, last_size)
        if step % self.ctrl_cfg.refine_interval == 0:
            self.refinement_after(step, optimizer)

    def after_train(
        self,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (radii > 0).flatten()
            full_mask = torch.zeros(self.num_points, device=radii.device, dtype=torch.bool)
            full_mask[self.filter_mask] = visible_mask
            
            grads = xys_grad.norm(dim=-1)
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=grads.device, dtype=grads.dtype)
                self.xys_grad_norm[self.filter_mask] = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[full_mask] = self.vis_counts[full_mask] + 1
                self.xys_grad_norm[full_mask] = grads[visible_mask] + self.xys_grad_norm[full_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros(self.num_points, device=radii.device, dtype=torch.float32)
            newradii = radii[visible_mask]
            self.max_2Dsize[full_mask] = torch.maximum(
                self.max_2Dsize[full_mask], newradii / float(last_size)
            )
        
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            self.class_prefix+"xyz": [self._means],
            self.class_prefix+"sh_dc": [self._features_dc],
            self.class_prefix+"sh_rest": [self._features_rest],
            self.class_prefix+"opacity": [self._opacities],
            self.class_prefix+"scaling": [self._scales],
            self.class_prefix+"rotation": [self._quats],
            self.class_prefix+"normal": [self._normals],
            self.class_prefix+"base_color": [self._base_color],
            self.class_prefix+"roughness": [self._roughness],
            self.class_prefix+"reflectivity": [self._reflectivity],
            self.class_prefix+"sun_visibility": [self._sun_visibility],
            # self.class_prefix+"incidents_dc": [self._incidents_dc],
            # self.class_prefix+"incidents_rest": [self._incidents_rest],
        }
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return self.get_gaussian_param_groups()

    def refinement_after(self, step, optimizer: torch.optim.Optimizer) -> None:
        assert step == self.step
        if self.step <= self.ctrl_cfg.warmup_steps:
            return
        with torch.no_grad():
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.ctrl_cfg.reset_alpha_interval
            do_densification = (
                self.step < self.ctrl_cfg.stop_split_at
                and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
            )
            # split & duplicate
            print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
            if do_densification:
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                
                avg_grad_norm = self.xys_grad_norm / self.vis_counts
                high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()
                
                splits = (
                    self.get_scaling.max(dim=-1).values > \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                if self.step < self.ctrl_cfg.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.ctrl_cfg.n_split_samples
                (
                    split_means,
                    split_feature_dc,
                    split_feature_rest,
                    split_opacities,
                    split_scales,
                    split_quats,
                    split_normals,
                    split_base_color,
                    split_roughness,
                    split_reflectivity,
                    split_sun_visibility,
                    split_incidents_dc,
                    split_incidents_rest,                    
                ) = self.split_gaussians(splits, nsamps)

                dups = (
                    self.get_scaling.max(dim=-1).values <= \
                        self.ctrl_cfg.densify_size_thresh * self.scene_scale
                ).squeeze()
                dups &= high_grads
                (
                    dup_means,
                    dup_feature_dc,
                    dup_feature_rest,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                    dup_normals, 
                    dup_base_color,
                    dup_roughness,
                    dup_reflectivity,
                    dup_sun_visibility,
                    dup_incidents_dc,
                    dup_incidents_rest,
                ) = self.dup_gaussians(dups)
                
                self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                # self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))
                self._features_dc = Parameter(torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0))
                self._features_rest = Parameter(torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0))
                self._opacities = Parameter(torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))

                self._normals = Parameter(torch.cat([self._normals.detach(), split_normals, dup_normals], dim=0))
                self._base_color = Parameter(torch.cat([self._base_color.detach(), split_base_color, dup_base_color], dim=0))
                self._roughness = Parameter(torch.cat([self._roughness.detach(), split_roughness, dup_roughness], dim=0))
                self._reflectivity = Parameter(torch.cat([self._reflectivity.detach(), split_reflectivity, dup_reflectivity], dim=0))
                self._sun_visibility = Parameter(torch.cat([self._sun_visibility.detach(), split_sun_visibility, dup_sun_visibility], dim=0))

                self._incidents_dc = Parameter(torch.cat([self._incidents_dc.detach(), split_incidents_dc, dup_incidents_dc], dim=0))
                self._incidents_rest = Parameter(torch.cat([self._incidents_rest.detach(), split_incidents_rest, dup_incidents_rest], dim=0))

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])],
                    dim=0,
                )
                
                split_idcs = torch.where(splits)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, split_idcs, param_groups, n=nsamps)

                dup_idcs = torch.where(dups)[0]
                param_groups = self.get_gaussian_param_groups()
                dup_in_optim(optimizer, dup_idcs, param_groups, 1)

            # cull NOTE: Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
            if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                deleted_mask = self.cull_gaussians()
                param_groups = self.get_gaussian_param_groups()
                remove_from_optim(optimizer, deleted_mask, param_groups)
            print(f"Class {self.class_prefix} left points: {self.num_points}")
                    
            # reset opacity
            if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                # NOTE: in nerfstudio, reset_value = cull_alpha_thresh * 0.8
                    # we align to original repo of gaussians spalting
                reset_value = torch.min(self.get_opacity.data,
                                        torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value)
                self._opacities.data = torch.logit(reset_value)
                # reset the exp of optimizer
                for group in optimizer.param_groups:
                    if group["name"] == self.class_prefix+"opacity":
                        old_params = group["params"][0]
                        param_state = optimizer.state[old_params]
                        param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                        param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None
        #self.update_visibility()

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            # cull huge ones
            toobigs = (
                torch.exp(self._scales).max(dim=-1).values > 
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())
        self._normals = Parameter(self._normals[~culls].detach())
        self._base_color = Parameter(self._base_color[~culls].detach())
        self._roughness = Parameter(self._roughness[~culls].detach())
        self._reflectivity = Parameter(self._reflectivity[~culls].detach())
        self._sun_visibility = Parameter(self._sun_visibility[~culls].detach())
        self._incidents_dc = Parameter(self._incidents_dc[~culls].detach())
        self._incidents_rest = Parameter(self._incidents_rest[~culls].detach())
        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int) -> Tuple:
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            self.get_scaling[split_mask].repeat(samps, 1) * centered_samples
            # torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self._scales[split_mask]) / size_fac).repeat(samps, 1)
        self._scales[split_mask] = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)

        new_normal = self._normals[split_mask].repeat(samps, 1)
        new_base_color = self._base_color[split_mask].repeat(samps, 1)
        new_roughenss = self._roughness[split_mask].repeat(samps, 1)
        new_reflectivity = self._reflectivity[split_mask].repeat(samps, 1)
        new_sun_visibility = self._sun_visibility[split_mask].repeat(samps, 1)

        new_incidents_dc = self._incidents_dc[split_mask].repeat(samps, 1, 1)
        new_incidents_rest = self._incidents_rest[split_mask].repeat(samps, 1, 1)

        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats, new_normal,new_base_color, new_roughenss, new_reflectivity, new_sun_visibility,new_incidents_dc, new_incidents_rest

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature_dc = self._features_dc[dup_mask]
        dup_feature_rest = self._features_rest[dup_mask]
        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]
        dup_normals = self._normals[dup_mask]
        dup_base_color = self._base_color[dup_mask]
        dup_roughness  = self._roughness[dup_mask]
        dup_reflectivity = self._reflectivity[dup_mask]
        dup_sun_visibility = self._sun_visibility[dup_mask]
        dup_incidents_dc  = self._incidents_dc[dup_mask]
        dup_incidents_rest  = self._incidents_rest[dup_mask]

        return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats , dup_normals, dup_base_color,dup_roughness, dup_reflectivity, dup_sun_visibility,dup_incidents_dc, dup_incidents_rest 

    def get_gaussians(self, cam: dataclass_camera) -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = self._means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
            
        activated_opacities = self.get_opacity
        activated_scales = self.get_scaling
        activated_rotations = self.get_quats
        activated_normals = self.get_normal
        activated_albedos = self.get_base_color
        activated_roughness = self.get_roughness
        activated_reflectivity = self.get_reflectivity
        activated_sun_visibility = self.get_sun_visibility
        activated_incidents = self.get_incidents
        actovated_colors = rgbs

        # collect gaussians information
        gs_dict = dict(
            _means=self._means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
            _normals = activated_normals[filter_mask],
            _albedos = activated_albedos[filter_mask],
            _roughness = activated_roughness[filter_mask],
            _reflectivity = activated_reflectivity[filter_mask],
            _sun_visibility = activated_sun_visibility[filter_mask],
            _incidents = activated_incidents[filter_mask],
            # _incident_dirs = self._incident_dirs[filter_mask],
            # _visibility_tracing = self._visibility_tracing[filter_mask],
            # _incident_areas = self._incident_areas[filter_mask]
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict
    
    def compute_reg_loss(self):
        loss_dict = {}
        sharp_shape_reg_cfg = self.reg_cfg.get("sharp_shape_reg", None)
        if sharp_shape_reg_cfg is not None:
            w = sharp_shape_reg_cfg.w
            max_gauss_ratio = sharp_shape_reg_cfg.max_gauss_ratio
            step_interval = sharp_shape_reg_cfg.step_interval
            if self.step % step_interval == 0:
                # scale regularization
                scale_exp = self.get_scaling
                scale_reg = torch.maximum(scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(max_gauss_ratio)) - max_gauss_ratio
                scale_reg = scale_reg.mean() * w
                loss_dict["sharp_shape_reg"] = scale_reg

        flatten_reg = self.reg_cfg.get("flatten", None)
        if flatten_reg is not None:
            sclaings = self.get_scaling
            min_scale, _ = torch.min(sclaings, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss = torch.abs(min_scale).mean()
            loss_dict["flatten"] = flatten_loss * flatten_reg.w
        
        sparse_reg = self.reg_cfg.get("sparse_reg", None)
        if sparse_reg:
            if (self.cur_radii > 0).sum():
                opacity = torch.sigmoid(self._opacities)
                opacity = opacity.clamp(1e-6, 1-1e-6)
                log_opacity = opacity * torch.log(opacity)
                log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
                sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[self.cur_radii > 0].mean()
                loss_dict["sparse_reg"] = sparse_loss * sparse_reg.w

        # compute the max of scaling
        max_s_square_reg = self.reg_cfg.get("max_s_square_reg", None)
        if max_s_square_reg is not None and not self.ball_gaussians:
            loss_dict["max_s_square"] = torch.mean((self.get_scaling.max(dim=1).values) ** 2) * max_s_square_reg.w
        return loss_dict
    
    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        N = state_dict["_means"].shape[0]
        self._means = Parameter(torch.zeros((N,) + self._means.shape[1:], device=self.device))
        self._scales = Parameter(torch.zeros((N,) + self._scales.shape[1:], device=self.device))
        self._quats = Parameter(torch.zeros((N,) + self._quats.shape[1:], device=self.device))
        self._features_dc = Parameter(torch.zeros((N,) + self._features_dc.shape[1:], device=self.device))
        self._features_rest = Parameter(torch.zeros((N,) + self._features_rest.shape[1:], device=self.device))
        self._opacities = Parameter(torch.zeros((N,) + self._opacities.shape[1:], device=self.device))
        self._normals = Parameter(torch.zeros((N,) + self._normals.shape[1:], device=self.device))
        self._base_color = Parameter(torch.zeros((N,) + self._base_color.shape[1:], device=self.device))
        self._roughness = Parameter(torch.zeros((N,) + self._roughness.shape[1:], device=self.device))
        self._reflectivity = Parameter(torch.zeros((N,) + self._reflectivity.shape[1:], device=self.device))
        self._sun_visibility = Parameter(torch.zeros((N,) + self._sun_visibility.shape[1:], device=self.device))
        self._incidents_dc = Parameter(torch.zeros((N,) + self._incidents_dc.shape[1:], device=self.device))
        self._incidents_rest = Parameter(torch.zeros((N,) + self._incidents_rest.shape[1:], device=self.device))
        msg = super().load_state_dict(state_dict, **kwargs)
        #self.update_visibility() 
        return msg
    
    def get_inverse_covariance(self, scaling_modifier=1):
        return self.covariance_activation(1 / self.get_scaling,
                                          1 / scaling_modifier,
                                          self.get_quats)

    def update_visibility(self, raytracer=None, sample_num=24,sun_direction=None):
        if raytracer is None:
            raytracer = RayTracer(self._means, self.get_scaling, self.get_quats)
        gaussians_xyz = self.get_xyz #self._means
        gaussians_inverse_covariance = self.get_inverse_covariance()
        gaussians_opacity = self.get_opacity[:, 0]
        gaussians_normal = self.get_normal

        # points = gaussians_xyz.detach().cpu().numpy()
        # colors = (-gaussians_normal+1).detach().cpu().numpy() * 128 
        # vertex_data = np.zeros(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        #                                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        # vertex_data['x'] = points[:, 0]
        # vertex_data['y'] = points[:, 1]
        # vertex_data['z'] = points[:, 2]
        # vertex_data['red'] = colors[:, 0]
        # vertex_data['green'] = colors[:, 1]
        # vertex_data['blue'] = colors[:, 2]

        # # Create a PlyElement
        # ply_element = PlyElement.describe(vertex_data, 'vertex')

        # # Save to a PLY file
        # ply_data = PlyData([ply_element])
        # ply_data.write('point_cloud.ply')

        # print("PLY file saved as 'point_cloud.ply'")


        incident_visibility_results = []
        incident_dirs_results = []
        incident_areas_results = []
        chunk_size = gaussians_xyz.shape[0] // ((sample_num - 1) // 24 + 1)
        for offset in tqdm(range(0, gaussians_xyz.shape[0], chunk_size), "Update visibility with raytracing."):
            incident_dirs, incident_areas = sample_incident_rays(gaussians_normal[offset:offset + chunk_size], True,
                                                    sample_num) #-1
            
            if sun_direction is None:
                sun_direction = torch.tensor([0.2, 1., 1.])
                sun_direction = sun_direction/sun_direction.norm()
                sun_direction = sun_direction.repeat(incident_dirs.shape[0],1,1).to(device=incident_dirs.device)
            else:
                sun_direction = sun_direction
                sun_direction = sun_direction/sun_direction.norm()
                sun_direction = sun_direction.repeat(incident_dirs.shape[0],1,1).to(device=incident_dirs.device)
            incident_dirs = torch.concat([sun_direction,incident_dirs], dim=1)
            incident_areas = torch.concat([incident_areas[:,0,:].unsqueeze(1),incident_areas], dim=1)
            

            trace_results = raytracer.trace_visibility(
                gaussians_xyz[offset:offset + chunk_size, None].expand_as(incident_dirs),
                incident_dirs,
                gaussians_xyz,
                gaussians_inverse_covariance,
                gaussians_opacity,
                gaussians_normal)
            incident_visibility = trace_results["visibility"]
            incident_visibility_results.append(incident_visibility)
            incident_dirs_results.append(incident_dirs)
            incident_areas_results.append(incident_areas)
        incident_visibility_result = torch.cat(incident_visibility_results, dim=0)
        incident_dirs_result = torch.cat(incident_dirs_results, dim=0)
        incident_areas_result = torch.cat(incident_areas_results, dim=0)
        self._visibility_tracing = incident_visibility_result.detach()
        self._incident_dirs = incident_dirs_result.detach()
        self._incident_areas = incident_areas_result.detach()
        #del raytracer

