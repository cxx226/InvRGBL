import os
import time
import random
from enum import IntEnum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
import logging
import kornia
import viser
import nerfview
from bvh import RayTracer
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .pbr import rendering_equation, rendering_equation_lidar
from models.gaussians.basics import *
from utils.graphics_utils import sample_incident_rays
from models.losses import normal_map_smooth_loss, region_consistency_loss, neighborhood_smoothness_loss

logger = logging.getLogger()



class GSModelType(IntEnum):
    Background = 0
    RigidNodes = 1
    SMPLNodes = 2
    DeformableNodes = 3

def lr_scheduler_fn(
    cfg: OmegaConf,
    lr_init: float
):
    if cfg.lr_final is None:
        lr_final = lr_init
    else:
        lr_final = cfg.lr_final

    def func(step):
        step = step - cfg.opt_after
        if step < 0:
            return 0.
        
        if step < cfg.warmup_steps:
            if cfg.ramp == "cosine":
                lr = cfg.lr_pre_warmup + (lr_init - cfg.lr_pre_warmup) * np.sin(
                    0.5 * np.pi * np.clip(step / cfg.warmup_steps, 0, 1)
                )
            else:
                lr = (
                    cfg.lr_pre_warmup
                    + (lr_init - cfg.lr_pre_warmup) * step / cfg.warmup_steps
                )
        else:
            t = np.clip(
                (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps), 0, 1
            )
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return lr  # divided by lr_init because the multiplier is with the initial learning rate

    return func

class BasicTrainer(nn.Module):
    def __init__(
        self,
        type: str = "basic",
        optim: OmegaConf = None,
        losses: OmegaConf = None,
        render: OmegaConf = None,
        res_schedule: OmegaConf = None,
        gaussian_optim_general_cfg: OmegaConf = None,
        gaussian_ctrl_general_cfg: OmegaConf = None,
        model_config: OmegaConf = None,
        num_train_images: int = 0,
        num_full_images: int = 0,
        test_set_indices: List[int] = None,
        scene_aabb: torch.Tensor = None,
        device=None,
        freeze_step: OmegaConf = None
    ):
        super().__init__()
        self._type = type
        self.optim_general = optim
        self.losses_dict = losses
        self.render_cfg = render
        self.res_schedule = res_schedule
        self.model_config = model_config
        self.num_iters = self.optim_general.get("num_iters", 30000)
        self.gaussian_optim_general_cfg = gaussian_optim_general_cfg
        self.gaussian_ctrl_general_cfg = gaussian_ctrl_general_cfg
        self.step = 0
        self.device = device

        self._visibility_tracings_list = {}
        self._incident_dirs_list = {}
        self._incident_areas_list = {}    
        self.labels = {}
        
        self.freeze_step = freeze_step 
        self.freezed = False

        # dataset infos
        self.num_train_images = num_train_images
        self.num_full_images = num_full_images
        
        # init scene scale
        self._init_scene(scene_aabb=scene_aabb)
        
        # init models
        self.models = {}
        self.misc_classes_keys = [
            'Sky', 'Affine', 'CamPose', 'CamPosePerturb'
        ]
        self.gaussian_classes = {}
        self._init_models()
        self.pts_labels = None # will be overwritten in forward
        self.render_dynamic_mask = False
        
        # init losses fn
        self._init_losses()
        
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        self.step = 0
        # background color
        self.back_color = torch.zeros(3).to(self.device)
        # for evaluation
        self.cur_frame = torch.tensor(0, device=self.device)
        self.test_set_indices = test_set_indices # will be override
        # a simple viewer for background visualization
        self.viewer = None
        self.pbr = self.render_cfg.pbr
        self.sun_intensity = 10
    
    @property
    def in_test_set(self):
        return self.cur_frame.item() in self.test_set_indices
    
    def set_train(self):
        for model in self.models.values():
            model.train()
        self.train()
    
    def set_eval(self):
        for model in self.models.values():
            model.eval()
        self.eval()

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max((self.res_schedule.downscale_times - self.step // self.res_schedule.double_steps), 0)
        else:
            return 1
        
    def update_gaussian_cfg(self, model_cfg: OmegaConf) -> OmegaConf:
        class_optim_cfg = model_cfg.get('optim', None)
        class_ctrl_cfg = model_cfg.get('ctrl', None)
        new_optim_cfg = self.gaussian_optim_general_cfg.copy()
        new_ctrl_cfg = self.gaussian_ctrl_general_cfg.copy()
        if class_optim_cfg is not None:
            new_optim_cfg.update(class_optim_cfg)
        if class_ctrl_cfg is not None:
            new_ctrl_cfg.update(class_ctrl_cfg)
        model_cfg['optim'] = new_optim_cfg
        model_cfg['ctrl'] = new_ctrl_cfg

        return model_cfg
        
    def _init_scene(self, scene_aabb) -> None:
        self.aabb = scene_aabb.to(self.device)
        scene_origin = (self.aabb[0] + self.aabb[1]) / 2
        scene_radius = torch.max(self.aabb[1] - self.aabb[0]) / 2 * 1.1
        self.scene_radius = scene_radius.item()
        self.scene_origin = scene_origin
        logger.info(f"scene origin: {scene_origin}")
        logger.info(f"scene radius: {scene_radius}")
    
    def _init_models(self) -> None:
        raise NotImplementedError("Please implement the _init_models function")
    
    def initialize_optimizer(self) -> None:
        # get param groups first
        self.param_groups = {}
        for class_name, model in self.models.items():
            self.param_groups.update(model.get_param_groups())
                 
        groups = []
        lr_schedulers = {}
        for params_name, params in self.param_groups.items():
            class_name = params_name.split("#")[0]
            component_name = params_name.split("#")[1]
            class_cfg = self.model_config.get(class_name)
            class_optim_cfg = class_cfg["optim"]
            
            raw_optim_cfg = class_optim_cfg.get(component_name, None)

            lr_scale_factor = raw_optim_cfg.get("scale_factor", 1.0)
            if isinstance(lr_scale_factor, str) and lr_scale_factor == "scene_radius":
                # scale the spatial learning rate to scene scale
                lr_scale_factor = self.scene_radius

            optim_cfg = OmegaConf.create({
                "lr": raw_optim_cfg.get('lr', 0.0005),
                "eps": raw_optim_cfg.get('eps', 1.0e-15),
                "weight_decay": raw_optim_cfg.get('weight_decay', 0),
            })
            optim_cfg.lr = optim_cfg.lr * lr_scale_factor
            assert optim_cfg is not None, f"param group {params_name} not found in config"
            lr_init = optim_cfg.lr
            groups.append({
                'params': params,
                'name': params_name,
                'lr': optim_cfg.lr,
                'eps': optim_cfg.eps,
                'weight_decay': optim_cfg.weight_decay
            })
            
            if raw_optim_cfg.get("lr_final", None) is not None:
                sched_cfg = OmegaConf.create({
                    "opt_after": raw_optim_cfg.get('opt_after', 0),
                    "warmup_steps": raw_optim_cfg.get('warmup_steps', 0),
                    "max_steps": raw_optim_cfg.get('max_steps', self.num_iters),
                    "lr_pre_warmup": raw_optim_cfg.get('lr_pre_warmup', 1.0e-8),
                    "lr_final": raw_optim_cfg.get('lr_final', None),
                    "ramp": raw_optim_cfg.get('ramp', "cosine"),
                })
                # scale the learning rate according to the scene scale
                sched_cfg.lr_pre_warmup = sched_cfg.lr_pre_warmup * lr_scale_factor
                sched_cfg.lr_final = sched_cfg.lr_final * lr_scale_factor if sched_cfg.lr_final is not None else None
                # adjust max_steps to account for opt_after
                sched_cfg.max_steps = sched_cfg.max_steps - sched_cfg.opt_after
                lr_schedulers[params_name] = lr_scheduler_fn(sched_cfg, lr_init)

        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.lr_schedulers = lr_schedulers
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.optim_general.get("use_grad_scaler", False))
        self.update_vis = False

    def update_visibility(self, update=False, sun_direction=None, mesh=False):
        if self.cur_frame.item() in self._visibility_tracings_list.keys():
            vis_num = self._visibility_tracings_list[self.cur_frame.item()].shape[0]
            pc_number = 0
            for class_name in self.gaussian_classes.keys():
                pc_number = pc_number + self.models[class_name].num_points
            if pc_number != vis_num:
                update = True

        if (self.cur_frame.item() not in self._visibility_tracings_list.keys()) or update:
            means = []
            scalings = []
            quats = []
            gaussians_inverse_covariance = []
            gaussians_opacity = []
            gaussians_normal = []
            with torch.no_grad():
                for class_name in self.gaussian_classes.keys():
                    means.append(self.models[class_name].get_xyz)
                    scalings.append(self.models[class_name].get_scaling)
                    quats.append(self.models[class_name].get_quats)
                    gaussians_inverse_covariance.append(self.models[class_name].get_inverse_covariance())
                    gaussians_opacity.append(self.models[class_name].get_opacity[:, 0])
                    gaussians_normal.append(self.models[class_name].get_normal)
                gaussians_xyz = means = torch.concat(means)
                scalings = torch.concat(scalings)
                quats = torch.concat(quats)
                raytracer  =  RayTracer(means, scalings, quats)
                gaussians_inverse_covariance = torch.concat(gaussians_inverse_covariance)
                gaussians_opacity = torch.concat(gaussians_opacity)
                gaussians_normal = torch.concat(gaussians_normal)
                

                incident_visibility_results = []
                incident_dirs_results = []
                incident_areas_results = []
                sample_num = 128 #24
                chunk_size = gaussians_xyz.shape[0]
                for offset in tqdm(range(0, gaussians_xyz.shape[0], chunk_size), "Update visibility with raytracing."):
                    incident_dirs, incident_areas = sample_incident_rays(gaussians_normal[offset:offset + chunk_size], True,
                                                            sample_num-1) #-1 TODO

                    if sun_direction is not None:
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
                del raytracer
                if self.cur_frame.item() in self._visibility_tracings_list:
                    del self._visibility_tracings_list[self.cur_frame.item()]
                    del self._incident_dirs_list[self.cur_frame.item()]
                    del self._incident_areas_list[self.cur_frame.item()]
                self._visibility_tracings_list.update({self.cur_frame.item(): incident_visibility_result.detach()})
                self._incident_dirs_list.update({self.cur_frame.item(): incident_dirs_result.detach()})
                self._incident_areas_list.update({self.cur_frame.item(): incident_areas_result.detach()}) 
        #del raytracer


    def reinitialize_optimizer(self,train_sky=False,train_incident=False,train_vis=False) -> None:
        # get param groups first
        self.param_groups = {}
        class_names = self.gaussian_classes.keys()
        if train_sky:
            class_name = 'Sky'
            model = self.models[class_name]
            self.param_groups.update(model.get_param_groups())
        if train_vis:
            for class_name in class_names:
                self.param_groups.update(
                    {self.models[class_name].class_prefix+"sun_visibility": [self.models[class_name]._sun_visibility],})
        if train_incident:
            for class_name in class_names:
                self.param_groups.update(
                    {self.models[class_name].class_prefix+"incidents_dc": [self.models[class_name]._incidents_dc],
                    self.models[class_name].class_prefix+"incidents_rest": [self.models[class_name]._incidents_rest],#})
                    self.models[class_name].class_prefix+"base_color": [self.models[class_name]._base_color],})

            # self.param_groups.update({
            #     self.models['Sky'].class_prefix+"sky_intensity_scale": [self.models['Sky'].sky_intensity_scale]
            # })

        groups = []
        lr_schedulers = {}
        for params_name, params in self.param_groups.items():
            class_name = params_name.split("#")[0]
            component_name = params_name.split("#")[1]
            class_cfg = self.model_config.get(class_name)
            class_optim_cfg = class_cfg["optim"]

            raw_optim_cfg = class_optim_cfg.get(component_name, None)

            lr_scale_factor = raw_optim_cfg.get("scale_factor", 1.0)
            if isinstance(lr_scale_factor, str) and lr_scale_factor == "scene_radius":
                # scale the spatial learning rate to scene scale
                lr_scale_factor = self.scene_radius

            optim_cfg = OmegaConf.create({
                "lr": raw_optim_cfg.get('lr', 0.0005),
                "eps": raw_optim_cfg.get('eps', 1.0e-15),
                "weight_decay": raw_optim_cfg.get('weight_decay', 0),
            })
            optim_cfg.lr = optim_cfg.lr * lr_scale_factor
            assert optim_cfg is not None, f"param group {params_name} not found in config"
            lr_init = optim_cfg.lr
            groups.append({
                'params': params,
                'name': params_name,
                'lr': optim_cfg.lr,
                'eps': optim_cfg.eps,
                'weight_decay': optim_cfg.weight_decay
            })
            
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        # self.lr_schedulers = lr_schedulers
        # self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.optim_general.get("use_grad_scaler", False))
    

    def _init_losses(self) -> None:
        sky_opacity_loss_fn = None
        if "Sky" in self.models:
            if self.losses_dict.mask.opacity_loss_type == "bce":
                from models.losses import binary_cross_entropy
                sky_opacity_loss_fn = lambda pred, gt: binary_cross_entropy(pred, gt, reduction="mean")
            elif self.losses_dict.mask.opacity_loss_type == "safe_bce":
                from models.losses import safe_binary_cross_entropy
                sky_opacity_loss_fn = lambda pred, gt: safe_binary_cross_entropy(pred, gt, limit=0.1, reduction="mean")
        self.sky_opacity_loss_fn = sky_opacity_loss_fn
        
        depth_loss_fn = None
        depth_loss_cfg = self.losses_dict.get("depth", None)
        if depth_loss_cfg is not None:
            from models.losses import DepthLoss
            depth_loss_fn = DepthLoss(
                loss_type=depth_loss_cfg.loss_type,
                normalize=depth_loss_cfg.normalize,
                use_inverse_depth=depth_loss_cfg.inverse_depth,
            )
        self.depth_loss_fn = depth_loss_fn
        self.region_consistency_loss = lambda a, b: region_consistency_loss(a, b) 
    
    def optimizer_zero_grad(self) -> None:
        #self.optimizer.zero_grad()
        self.optimizer.zero_grad(set_to_none=True) 
    
    def optimizer_step(self) -> None:
        self.optimizer.step()

    def preprocess_per_train_step(self, step: int) -> None:
        self.step = step
        for class_name in self.gaussian_classes.keys():
            self.models[class_name].preprocess_per_train_step(step)

        # viewer
        if self.viewer is not None:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            self.tic = time.time()
        
    def postprocess_per_train_step(self, step: int) -> None:
        if step < self.freeze_step:   
            radii = self.info["radii"]
            if self.render_cfg.absgrad:
                grads = self.info["means2d"].absgrad.clone()
            else:
                grads = self.info["means2d"].grad.clone()
            if len(grads.shape)<3:
                grads = grads.unsqueeze(0)
            
            grads[..., 0] *= self.info["width"] / 2.0 * self.render_cfg.batch_size
            grads[..., 1] *= self.info["height"] / 2.0 * self.render_cfg.batch_size
            
            for class_name in self.gaussian_classes.keys():
                gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
                self.models[class_name].postprocess_per_train_step(
                    step=step,
                    optimizer=self.optimizer,
                    radii=radii[0, gaussian_mask],
                    xys_grad=grads[0, gaussian_mask],
                    last_size=max(self.info["width"], self.info["height"])
                )

            del grads

        else: 
            if not self.freezed:
                print('freeze GS parameters, focus on lighting more')
                for class_name in self.gaussian_classes.keys():
                    gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
                    for param in self.models[class_name].parameters():
                        param.requires_grad = False 
                self.reinitialize_optimizer(train_sky=True)
                self.freezed = True
            
        # viewer
        if self.viewer is not None:
            num_train_rays_per_step = self.render_cfg.batch_size * self.info["width"] * self.info["height"]
            self.viewer.lock.release()
            num_train_steps_per_sec = 1.0 / (time.time() - self.tic)
            num_train_rays_per_sec = (
                num_train_rays_per_step * num_train_steps_per_sec
            )
            # Update the viewer state.
            self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            # Update the scene.
            self.viewer.update(step, num_train_rays_per_step)
    
    def update_visibility_filter(self) -> None:
        for class_name in self.gaussian_classes.keys():
            gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
            self.models[class_name].cur_radii = self.info["radii"][0, gaussian_mask]

    def process_camera(
        self,
        camera_infos: Dict[str, torch.Tensor],
        image_ids: torch.Tensor,
        novel_view: bool = False
    ) -> dataclass_camera:
        camtoworlds = camtoworlds_gt = camera_infos["camera_to_world"]
        
        if "CamPosePerturb" in self.models.keys() and not novel_view:
            camtoworlds = self.models["CamPosePerturb"](camtoworlds, image_ids)

        if "CamPose" in self.models.keys() and not novel_view:
            camtoworlds = self.models["CamPose"](camtoworlds, image_ids)
        
        # collect camera information
        camera_dict = dataclass_camera(
            camtoworlds=camtoworlds,
            camtoworlds_gt=camtoworlds_gt,
            Ks=camera_infos["intrinsics"],
            H=camera_infos["height"],
            W=camera_infos["width"]
        )
        
        return camera_dict

    def collect_gaussians(
        self,
        cam: dataclass_camera,
        image_ids: torch.Tensor, # leave it here for future use
        sun_direction = None,
        update = False
    ) -> dataclass_gs:
        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
            "class_labels": [],
        }
        
        if (self.step > self.freeze_step) and (self.step % 100 == 1):
            update = random.random() < 0.01   
        
        self.update_visibility(update=update, sun_direction=sun_direction)


        if self.pbr:
            gs_dict.update({"_normals":[], "_albedos":[],"_roughness":[],"_reflectivity":[],"_sun_visibility":[]})
            gs_dict.update({"_incidents":[]}) 

        for class_name in self.gaussian_classes.keys():
            gs = self.models[class_name].get_gaussians(cam)
            if gs is None:
                continue
            # collect gaussians
            gs["class_labels"] = torch.full((gs["_means"].shape[0],), self.gaussian_classes[class_name], device=self.device)
            for k, _ in gs.items():
                #if k == "_normals":
                gs_dict[k].append(gs[k])
                    
        
        for k, v in gs_dict.items():
            gs_dict[k] = torch.cat(v, dim=0)
            
        # get the class labels
        self.pts_labels = gs_dict.pop("class_labels")
        if self.render_dynamic_mask:
            self.dynamic_pts_mask = (self.pts_labels != 0).float()

        extras = None
        if self.pbr:
            extras = {'normals':gs_dict["_normals"],"albedos":gs_dict["_albedos"],"roughness":gs_dict["_roughness"], "reflectivity":gs_dict["_reflectivity"], "_incidents":gs_dict["_incidents"]}
            extras.update({
                '_incident_dirs':self._incident_dirs_list[self.cur_frame.item()],
                            '_visibility_tracing':self._visibility_tracings_list[self.cur_frame.item()], #list(self._visibility_tracings_list.values())
                            '_incident_areas':self._incident_areas_list[self.cur_frame.item()],
                "sun_visibility":gs_dict["_sun_visibility"]})


        gaussians = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            detach_keys=[],    # if "means" in detach_keys, then the means will be detached
            extras=extras        # to save some extra information (TODO) more flexible way
        )
        
        return gaussians
    
    def render_gaussians(
        self,
        gs: dataclass_gs,
        cam: dataclass_camera,
        direct_light_env_light = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        def render_fn(opaticy_mask=None, return_info=False):
            #if gs == type? 
            color_feature = gs.rgbs
            if self.pbr:
                cam_pos = cam.camtoworlds[None, :3, 3] 
                cam_pos = cam_pos 
                viewdirs = F.normalize(cam_pos - gs.means, dim=-1)
                view_dists = torch.norm(cam_pos - gs.means, dim=1) 
                view_dists = torch.sigmoid(view_dists/10)
                view_dists = view_dists[...,None]
                normals = gs.extras['normals']
                albedos = gs.extras['albedos']
                roughness = gs.extras['roughness']
                reflectivity = gs.extras['reflectivity']
                sun_visibility = gs.extras['sun_visibility']
                incidents = gs.extras['_incidents']

                color_feature = torch.cat([color_feature, normals, albedos, roughness,reflectivity,sun_visibility], dim=-1)

                #assert direct_light_env_light is not None
                intensity = rendering_equation_lidar(reflectivity,roughness,normals.detach(), viewdirs,view_dists.detach())

                brdf_color, extra_results = rendering_equation(
                    base_color = albedos, roughness = roughness, normals = normals.detach(), viewdirs = viewdirs,
                    incidents = incidents,
                    direct_light_env_light = direct_light_env_light,
                    incident_dirs = gs.extras['_incident_dirs'],
                    incident_areas = gs.extras['_incident_areas'],
                    visibility_precompute = gs.extras['_visibility_tracing'], 
                    sun_visibility = sun_visibility, #self.sun_visibility, #,
                    xyz = gs.means,
                    step = self.step,
                    )
                diffuse_light = extra_results["diffuse_light"]
                incident_sun_light = extra_results["incident_sun_light"]
                color_feature = torch.cat([color_feature, intensity, brdf_color, diffuse_light, incident_sun_light], dim=-1)


            renders, alphas, info = rasterization(
                means=gs.means,
                quats=gs.quats,
                scales=gs.scales,
                opacities=gs.opacities.squeeze()*opaticy_mask if opaticy_mask is not None else gs.opacities.squeeze(),
                colors=color_feature,
                viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
                Ks=cam.Ks[None, ...],  # [C, 3, 3]
                width=cam.W,
                height=cam.H,
                packed=self.render_cfg.packed,
                absgrad=self.render_cfg.absgrad,
                sparse_grad=self.render_cfg.sparse_grad,
                rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                **kwargs,
            )
            renders = renders[0]
            alphas = alphas[0].squeeze(-1)
            assert self.render_cfg.batch_size == 1, "batch size must be 1, will support batch size > 1 in the future"
            
            if self.pbr:
                rendered_rgb,rendered_normal,rendered_albedos,rendered_roughness, rendered_reflectivity, \
                rendered_sun_visibility, rendered_intensity, rendered_pbr, diffuse_light, incident_sun_light, rendered_depth = \
                 torch.split(renders, [3,3,3,1,1,1,1,3,3,3,1], dim=-1)              
                info.update({'rendered_normal':rendered_normal,
                            'rendered_albedos':rendered_albedos,
                            'rendered_roughness':rendered_roughness,
                            'rendered_reflectivity':rendered_reflectivity,
                            'rendered_pbr': rendered_pbr,
                            'diffuse_light':diffuse_light,
                            'rendered_intensity':rendered_intensity,
                            'rendered_sun_visibility':rendered_sun_visibility,
                            'incident_sun_light':incident_sun_light,
                            })

            else:
                assert renders.shape[-1] == 4, f"Must render rgb, depth and alpha"
                rendered_rgb, rendered_depth = torch.split(renders, [3, 1], dim=-1)
            
            if not return_info:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None]
            else:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None], info
        
        # render rgb and opacity
        rgb, depth, opacity, self.info = render_fn(return_info=True)
        results = {
            "rgb_gaussians": rgb,
            "depth": depth, 
            "opacity": opacity,
            #"normal": self.info['normal']
        }
        if self.pbr:
            results.update({'rendered_normal':self.info['rendered_normal'],
            'rendered_albedos':self.info['rendered_albedos'],
            'rendered_roughness':self.info['rendered_roughness'],
            'rendered_reflectivity':self.info['rendered_reflectivity'],
            'rendered_pbr':self.info['rendered_pbr'],
            'rendered_intensity':self.info['rendered_intensity'],
            'diffuse_light':self.info['diffuse_light'],
            'rendered_sun_visibility': self.info['rendered_sun_visibility'],
            'incident_sun_light':self.info['incident_sun_light'],
            })
        
        if self.training:
            self.info["means2d"].retain_grad()
        
        return results, render_fn


    def affine_transformation(
        self,
        rgb_blended: torch.Tensor,
        image_infos: Dict[str, torch.Tensor]
        ):
        if "Affine" in self.models:
            affine_trs = self.models['Affine'](image_infos)
            rgb_transformed = (affine_trs[..., :3, :3] @ rgb_blended[..., None] + affine_trs[..., :3, 3:])[..., 0]
            
            return rgb_transformed
        else:       
            return rgb_blended
    
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
            novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set
        
        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )
        gs = self.collect_gaussians(
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0]
        )

        # render gaussians
        outputs, _ = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.),
        )
        
        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
        )
        
        return outputs
    
    def backward(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        # ----------------- backward ----------------
        total_loss = sum(loss for loss in loss_dict.values())
        self.grad_scaler.scale(total_loss).backward()
        self.optimizer_step()
        
        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            for group in self.optimizer.param_groups:
                if group["name"] in self.lr_schedulers:
                    new_lr = self.lr_schedulers[group["name"]](self.step)
                    group["lr"] = new_lr
                
    def get_loss_weight(self, loss_name, default=1.0):
        """从 config 中获取 loss 的权重，没有就返回 default"""
        loss_cfg = self.losses_dict.get(loss_name, None)
        if loss_cfg is None:
            return default
        if isinstance(loss_cfg, dict):
            return loss_cfg.get("w", default)
        elif hasattr(loss_cfg, "w"):
            return loss_cfg.w
        return loss_cfg


    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # calculate loss
        loss_dict = {}

        if "egocar_masks" in image_infos:
            valid_loss_mask = (1.0 - image_infos["egocar_masks"]).float()
        else:
            valid_loss_mask = torch.ones_like(image_infos["sky_masks"])

        # ------------------------------
        # After freeze step
        # ------------------------------
        if self.step > self.freeze_step:
            if "rendered_pbr" in outputs:
                rendered_pbr = outputs["rendered_pbr"] * valid_loss_mask[..., None]
                gt_rgb = image_infos["pixels"] * valid_loss_mask[..., None]
                Ll1_pbr = torch.abs(rendered_pbr - gt_rgb).mean()
                loss_dict["pbr_loss"] = self.get_loss_weight("pbr") * Ll1_pbr

            if "diffuse_light" in outputs:
                diffuse_light = outputs["diffuse_light"]
                mean_light = diffuse_light.mean(-1, keepdim=True).expand_as(diffuse_light)
                loss_light = F.l1_loss(diffuse_light, mean_light)
                loss_dict["diffuse_light_loss"] = self.get_loss_weight("diffuse_light") * loss_light

            if "albedo_images" in image_infos:
                albedo_images_mask = (1 - image_infos["sky_masks"][..., None]) * valid_loss_mask[..., None]
                gt_albedo = image_infos["albedo_images"] * albedo_images_mask
                predicted_albedo = outputs["rendered_albedos"] * albedo_images_mask
                Ll1_albedo = torch.abs(gt_albedo - predicted_albedo).mean()
                loss_dict["albedo_loss"] = self.get_loss_weight("albedo") * Ll1_albedo

        # ------------------------------
        # Before freeze step
        # ------------------------------
        else:
            if "rendered_pbr" in outputs:
                rendered_pbr = outputs["rendered_pbr"] * valid_loss_mask[..., None]
                gt_rgb = image_infos["pixels"] * valid_loss_mask[..., None]
                Ll1_pbr = torch.abs(rendered_pbr - gt_rgb).mean()
                loss_dict["pbr_loss"] = self.get_loss_weight("pbr_pre") * Ll1_pbr

            if "roughness_images" in image_infos:
                roughness_images_mask = (1 - image_infos["sky_masks"][..., None]) * valid_loss_mask[..., None]
                rendered_roughness = outputs["rendered_roughness"] * roughness_images_mask
                gt_roughness = image_infos["roughness_images"] * roughness_images_mask
                L1_rough_loss = torch.abs(gt_roughness - rendered_roughness).mean()
                loss_dict["roughness_loss"] = self.get_loss_weight("roughness") * L1_rough_loss

                smooth_roughness_loss = normal_map_smooth_loss(rendered_roughness[None,...])
                loss_dict["smooth_roughness_loss"] = self.get_loss_weight("smooth_roughness") * smooth_roughness_loss

            if "shading_images" in image_infos:
                images_mask = (1 - image_infos["sky_masks"][..., None]) * valid_loss_mask[..., None]
                rendered_sun_visibility = outputs["rendered_sun_visibility"] * images_mask
                gt_sun_visibility = image_infos["shading_images"] * images_mask

                smooth_sun_visibility_loss = normal_map_smooth_loss(rendered_sun_visibility[None,...])
                loss_dict["smooth_sun_visibility_loss"] = self.get_loss_weight("smooth_sun_visibility") * smooth_sun_visibility_loss

                sun_visibility_loss = torch.abs(gt_sun_visibility - rendered_sun_visibility).mean()
                loss_dict["sun_visibility_loss"] = self.get_loss_weight("sun_visibility") * sun_visibility_loss

            if "intensity_images" in image_infos: 
                intensity_images_mask = valid_loss_mask[..., None] * (image_infos['intensity_images'] > 1e-3) 
                rendered_intensity = outputs["rendered_intensity"] * intensity_images_mask
                intensity_images = image_infos['intensity_images'] * intensity_images_mask
                Ll1_intensity = torch.abs(rendered_intensity - intensity_images).mean()
                loss_dict["intensity_loss"] = self.get_loss_weight("intensity") * Ll1_intensity

                images_mask = (1 - image_infos["sky_masks"][..., None]) * valid_loss_mask[..., None]
                rendered_reflectivity = outputs["rendered_reflectivity"] * images_mask
                gt_albedo = image_infos["albedo_images"] * images_mask

                albedo_mean = gt_albedo.mean(dim=-1)
                ref_neighborhood_smoothness_loss = neighborhood_smoothness_loss(albedo_mean[...,None], rendered_reflectivity)
                loss_dict["ref_neighborhood_smoothness_loss"] = self.get_loss_weight("ref_neighborhood_smoothness") * ref_neighborhood_smoothness_loss

                # ref_region_consistency_loss = self.region_consistency_loss(gt_albedo, rendered_reflectivity)
                # loss_dict["ref_region_consistency_loss"] = self.get_loss_weight("ref_region_consistency") * ref_region_consistency_loss

            if "albedo_images" in image_infos:
                albedo_images_mask = (1 - image_infos["sky_masks"][..., None]) * valid_loss_mask[..., None]
                gt_albedo = image_infos["albedo_images"] * albedo_images_mask
                predicted_albedo = outputs["rendered_albedos"] * albedo_images_mask

                albedo_smooth_loss = normal_map_smooth_loss(predicted_albedo[None,...])
                loss_dict["albedo_smooth_loss"] = self.get_loss_weight("albedo_smooth") * albedo_smooth_loss

                ref_region_consistency_loss = self.region_consistency_loss(rendered_reflectivity.detach(), predicted_albedo.mean(dim=-1))
                loss_dict["ref_region_consistency_loss"] = self.get_loss_weight("ref_region_consistency_albedo") * ref_region_consistency_loss

                Ll1_albedo = torch.abs(gt_albedo - predicted_albedo).mean()
                loss_dict["albedo_loss"] = self.get_loss_weight("albedo_pre") * Ll1_albedo

            if "normal_images" in image_infos:
                normal_images_mask = (1 - image_infos["sky_masks"][..., None]) * valid_loss_mask[..., None]
                normal_images = image_infos['normal_images'] * normal_images_mask
                predicted_normal = outputs["rendered_normal"] * normal_images_mask

                Ll1_normal_sparse = torch.abs(predicted_normal - normal_images).mean()
                loss_dict["gt_normal_loss"] = self.get_loss_weight("gt_normal") * Ll1_normal_sparse

                smooth_normal_loss = normal_map_smooth_loss(predicted_normal[None,...])
                loss_dict["smooth_normal_loss"] = self.get_loss_weight("smooth_normal") * smooth_normal_loss

            # rgb loss
            gt_rgb = image_infos["pixels"] * valid_loss_mask[..., None]
            predicted_rgb = outputs["rgb"] * valid_loss_mask[..., None]

            Ll1 = torch.abs(gt_rgb - predicted_rgb).mean()
            simloss = 1 - self.ssim(gt_rgb.permute(2, 0, 1)[None, ...],
                                    predicted_rgb.permute(2, 0, 1)[None, ...])
            loss_dict["rgb_loss"] = self.get_loss_weight("rgb") * Ll1
            loss_dict["ssim_loss"] = self.get_loss_weight("ssim") * simloss

            # mask loss
            gt_occupied_mask = (1.0 - image_infos["sky_masks"]).float() * valid_loss_mask
            pred_occupied_mask = outputs["opacity"].squeeze() * valid_loss_mask
            if self.sky_opacity_loss_fn is not None:
                sky_loss_opacity = self.sky_opacity_loss_fn(pred_occupied_mask, gt_occupied_mask)
                loss_dict["sky_loss_opacity"] = self.get_loss_weight("mask") * sky_loss_opacity

            # depth loss
            if self.depth_loss_fn is not None:
                gt_depth = image_infos["lidar_depth_map"] 
                lidar_hit_mask = (gt_depth > 0).float() * valid_loss_mask
                pred_depth = outputs["depth"]

                if self.step < self.freeze_step:
                    depth_loss = self.depth_loss_fn(pred_depth, gt_depth, lidar_hit_mask)
                    lidar_w_decay = self.losses_dict.depth.get("lidar_w_decay", -1)
                    if lidar_w_decay > 0:
                        decay_weight = np.exp(-self.step / 8000 * lidar_w_decay)
                    else:
                        decay_weight = 1
                    depth_loss = self.get_loss_weight("depth") * depth_loss * decay_weight
                    loss_dict["depth_loss"] = depth_loss

            # opacity entropy reg
            if "opacity_entropy" in self.losses_dict:
                pred_opacity = torch.clamp(outputs["opacity"].squeeze(), 1e-6, 1 - 1e-6)
                loss_dict["opacity_entropy_loss"] = self.get_loss_weight("opacity_entropy") * (
                    -pred_opacity * torch.log(pred_opacity)
                ).mean()

            # inverse depth smoothness reg
            if "inverse_depth_smoothness" in self.losses_dict:
                inverse_depth = 1 / (outputs["depth"] + 1e-5)
                loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(
                    inverse_depth[None].repeat(1, 1, 1, 3).permute(0, 3, 1, 2),
                    image_infos["pixels"][None].permute(0, 3, 1, 2)
                )
                loss_dict["inverse_depth_smoothness_loss"] = self.get_loss_weight("inverse_depth_smoothness") * loss_inv_depth

            # affine reg
            if "affine" in self.losses_dict and "Affine" in self.models:
                affine_trs = self.models['Affine']({"img_idx": image_infos["img_idx"].flatten()[0]})
                reg_mat = torch.eye(3, device=self.device)
                reg_shift = torch.zeros(3, device=self.device)
                loss_affine = torch.abs(affine_trs[..., :3, :3] - reg_mat).mean() \
                            + torch.abs(affine_trs[..., :3, 3:] - reg_shift).mean()
                loss_dict["affine_loss"] = self.get_loss_weight("affine") * loss_affine

            # dynamic region loss
            dynamic_region_cfg = self.losses_dict.get("dynamic_region", None)
            if dynamic_region_cfg is not None:
                weight_factor = dynamic_region_cfg.get("w", 1.0)
                start_from = dynamic_region_cfg.get("start_from", 0)
                if self.step == start_from:
                    self.render_dynamic_mask = True
                if self.step > start_from and "Dynamic_opacity" in outputs:
                    dynamic_pred_mask = (outputs["Dynamic_opacity"].data > 0.2).squeeze()
                    dynamic_pred_mask = dynamic_pred_mask & valid_loss_mask.bool()
                    if dynamic_pred_mask.sum() > 0:
                        Ll1 = torch.abs(gt_rgb[dynamic_pred_mask] - predicted_rgb[dynamic_pred_mask]).mean()
                        loss_dict["vehicle_region_rgb_loss"] = weight_factor * Ll1

            # gaussian reg losses
            for class_name in self.gaussian_classes.keys():
                class_reg_loss = self.models[class_name].compute_reg_loss()
                for k, v in class_reg_loss.items():
                    loss_dict[f"{class_name}_{k}"] = v

        return loss_dict

    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        metric_dict = {}
        psnr = self.psnr(outputs["rgb"], image_infos["pixels"])
        metric_dict.update({"psnr": psnr})
        return metric_dict
    
    def get_gaussian_count(self):
        num_dict = {}
        for class_name in self.gaussian_classes.keys():
            num_dict[class_name] = self.models[class_name].num_points
        return num_dict
    
    def state_dict(self, only_model: bool = True):
        state_dict = super().state_dict()
        state_dict.update({
            "models": {k: v.state_dict() for k, v in self.models.items()},
            "step": self.step,
        })
        if not only_model:
            state_dict.update({
                "optimizer": {k: v.state_dict() for k, v in self.optimizer.items()},
            })
        return state_dict

    def load_state_dict(self, state_dict: dict, load_only_model: bool =True, strict: bool = True):
        step = state_dict.pop("step")
        self.step = step
        logger.info(f"Loading checkpoint at step {step}")

        # load optimizer and schedulers
        if "optimizer" in state_dict:
            loaded_state_optimizers = state_dict.pop("optimizer")
        # if "schedulers" in state_dict:
        #     loaded_state_schedulers = state_dict.pop("schedulers")
        # if "grad_scaler" in state_dict:
        #     loaded_grad_scaler = state_dict.pop("grad_scaler")
        if not load_only_model:
            raise NotImplementedError("Now only support loading model, \
                it seems there is no need to load optimizer and schedulers")
            for k, v in loaded_state_optimizers.items():
                self.optimizer[k].load_state_dict(v)
            for k, v in loaded_state_schedulers.items():
                self.schedulers[k].load_state_dict(v)
            self.grad_scaler.load_state_dict(loaded_grad_scaler)
        
        # load model
        model_state_dict = state_dict.pop("models")
        for class_name in self.models.keys():
            # if class_name == 'Sky': #TODO
            #     continue
            model = self.models[class_name]
            model.step = step
            if class_name not in model_state_dict:
                if class_name in self.gaussian_classes:
                    self.gaussian_classes.pop(class_name)
                logger.warning(f"Cannot find {class_name} in the checkpoint")
                continue
            msg = model.load_state_dict(model_state_dict[class_name], strict=strict)
            logger.info(f"{class_name}: {msg}")
        msg = super().load_state_dict(state_dict, strict)
        logger.info(f"BasicTrainer: {msg}")
        
    def resume_from_checkpoint(
        self,
        ckpt_path: str,
        load_only_model: bool=True
    ) -> None:
        """
        Load model from checkpoint.
        """
        logger.info(f"Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path,weights_only=False)
        self.load_state_dict(state_dict, load_only_model=load_only_model, strict=False)
        
    def save_checkpoint(
        self,
        log_dir: str,
        save_only_model: bool=True,
        is_final: bool=False
    ) -> None:
        """
        Save model to checkpoint.
        """
        if is_final:
            ckpt_path = os.path.join(log_dir, f"checkpoint_final.pth")
        else:
            ckpt_path = os.path.join(log_dir, f"checkpoint_{self.step:05d}.pth")
        torch.save(self.state_dict(only_model=save_only_model), ckpt_path)
        logger.info(f"Saved a checkpoint to {ckpt_path}")
        
    def init_viewer(self, port: int = 8080):
        # a simple viewer for background ONLY visualization
        self.server = viser.ViserServer(port=port, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            mode="training",
        )

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)
        
        cam = dataclass_camera(
            camtoworlds=c2w,
            camtoworlds_gt=c2w,
            Ks=K,
            H=H,
            W=W
        )
        
        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
        }
        for class_name in ["Background"]:
            gs = self.models[class_name].get_gaussians(cam)
            if gs is None:
                continue

            for k, _ in gs.items():
                gs_dict[k].append(gs[k])
        
        for k, v in gs_dict.items():
            gs_dict[k] = torch.cat(v, dim=0)

        gs = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            detach_keys=[],
            extras=None
        )
        
        render_colors, _, _ = rasterization(
            means=gs.means,
            quats=gs.quats,
            scales=gs.scales,
            opacities=gs.opacities.squeeze(),
            colors=gs.rgbs,
            viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
            Ks=cam.Ks[None, ...],  # [C, 3, 3]
            width=cam.W,
            height=cam.H,
            packed=self.render_cfg.packed,
            absgrad=self.render_cfg.absgrad,
            sparse_grad=self.render_cfg.sparse_grad,
            rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
            radius_clip=4.0,  # skip GSs that have small image radius (in pixels)
        )
        return render_colors[0].cpu().numpy()