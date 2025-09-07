import torch
from utils.graphics_utils import sample_incident_rays
import torch.nn.functional as F
import numpy as np

def GGX_specular(
        normal,
        pts2c,
        pts2l,
        roughness,
        fresnel
):
    L = F.normalize(pts2l, dim=-1)  # [nrays, nlights, 3]
    V = F.normalize(pts2c, dim=-1)  # [nrays, 3]
    H = F.normalize((L + V[:, None, :]) / 2.0, dim=-1)  # [nrays, nlights, 3]
    N = F.normalize(normal, dim=-1)  # [nrays, 3]

    NoV = torch.sum(V * N, dim=-1, keepdim=True)  # [nrays, 1]
    N = N * NoV.sign()  # [nrays, 3]

    NoL = torch.sum(N[:, None, :] * L, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1] TODO check broadcast
    NoV = torch.sum(N * V, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, 1]
    NoH = torch.sum(N[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]
    VoH = torch.sum(V[:, None, :] * H, dim=-1, keepdim=True).clamp_(1e-6, 1)  # [nrays, nlights, 1]

    alpha = roughness * roughness  # [nrays, 3]
    alpha2 = alpha * alpha  # [nrays, 3]
    k = (alpha + 2 * roughness + 1.0) / 8.0
    FMi = ((-5.55473) * VoH - 6.98316) * VoH
    frac0 = fresnel + (1 - fresnel) * torch.pow(2.0, FMi)  # [nrays, nlights, 3]
    
    frac = frac0 * alpha2[:, None, :]  # [nrays, 1]
    nom0 = NoH * NoH * (alpha2[:, None, :] - 1) + 1

    nom1 = NoV * (1 - k) + k
    nom2 = NoL * (1 - k[:, None, :]) + k[:, None, :]
    nom = (4 * np.pi * nom0 * nom0 * nom1[:, None, :] * nom2).clamp_(1e-6, 4 * np.pi)
    spec = frac / nom
    return spec



### USING ###
def rendering_equation_lidar(base_color, roughness, normals, viewdirs, view_dists):
    #### neglecting d^2 in rendering equation
    normals = normals.detach()
    #roughness = roughness.detach()
    
    n_d_i = (normals * viewdirs).sum(-1, keepdim=True).clamp(min=0)
    f_d = base_color[:, None] #/ np.pi
    f_s = 0 #GGX_specular(normals, viewdirs, viewdirs[:,None, :], roughness, fresnel=0.04)

    transport =  1 #n_d_i[:,None,:]  # （num_pts, num_sample, 1)
    #specular = ((f_s) * transport).mean(dim=-2)
    pbr = ((f_d + f_s) * transport).mean(dim=-2) / (view_dists **2)
    return pbr



### USING ###
def rendering_equation(base_color, roughness, normals, viewdirs,
                              incidents=None, direct_light_env_light=None,
                              incident_dirs=None, incident_areas=None, visibility_precompute=None,sample_num=24,sun_visibility=None,xyz=None,step=None):
    
    normals = normals.detach()
    # if incident_dirs is None:
    #     incident_dirs, incident_areas = sample_incident_rays(normals, True, sample_num)
    with_sun = True
    if with_sun: #else:
        sun_visibility = visibility_precompute[:,0,:]
        sun_direction = incident_dirs[:,0,:]
        incident_areas = incident_areas[:,1:,:]
        visibility_precompute = visibility_precompute[:,1:,:]
        incident_dirs = incident_dirs[:,1:,:]
    else:        
        sun_direction = direct_light_env_light.get_sun_direction()
        sun_direction = sun_direction[None,...].repeat(incident_dirs.shape[0],1)


    deg = int(np.sqrt(incidents.shape[1]) - 1)
    

    global_incident_lights = direct_light_env_light.direct_light(incident_dirs,step) #)

    global_incident_lights = torch.ones_like(global_incident_lights) * torch.tensor([200, 200, 180]).to(device=normals.device) / 255 * 1.5
    local_incident_lights = 0 #eval_sh(deg, incidents.transpose(1, 2).view(-1, 1, 3, (deg + 1) ** 2), incident_dirs).clamp_min(0)
    incident_visibility = visibility_precompute
    # incident_visibility[incident_visibility>0.5] = 1
    # incident_visibility[incident_visibility<=0.5] = 0
    global_incident_lights = global_incident_lights * incident_visibility
    incident_lights = global_incident_lights + local_incident_lights  

    n_d_i = (normals[:, None] * incident_dirs).sum(-1, keepdim=True).clamp(min=0)
    f_d = base_color[:, None] / np.pi
    f_s = 0 #GGX_specular(normals, viewdirs, incident_dirs, roughness, fresnel=0.04)

    transport = incident_lights * incident_areas * n_d_i  # （num_pts, num_sample, 3)
    

    specular = ((f_s) * transport).mean(dim=-2)
    pbr = ((f_d + f_s) * transport).mean(dim=-2)
    diffuse_light = transport.mean(dim=-2)

    if with_sun and (sun_visibility is not None):
        sun_visibility = sun_visibility.detach()
        #sun_visibility = torch.where(sun_visibility < 0.95, torch.tensor(0.0), sun_visibility)
        intensity = direct_light_env_light.sun_intensity[None,...].repeat(incident_dirs.shape[0],1) #* 0.5
        #intensity = torch.ones_like(intensity) *torch.tensor([255, 178, 102]).to(device=intensity.device) * 3 / 255 #* 3
        sun_light = (sun_direction * normals).sum(dim=-1)[...,None] * intensity * 3#torch.ones_like(intensity) * 3 # * 10
        sun_light = sun_light.clip(0,10)
        f_d_ = (base_color / np.pi)
        f_s_ = GGX_specular(normals, viewdirs, sun_direction.unsqueeze(-2).detach(), roughness, fresnel=0.04).squeeze(-2)
        incident_sun_light = (f_d_ + f_s_) * sun_light * 0.3 
        pbr = pbr + incident_sun_light * sun_visibility 

    pbr = pbr 

    extra_results = {
        "incident_dirs": incident_dirs,
        "incident_lights": incident_lights,
        "local_incident_lights": local_incident_lights,
        "global_incident_lights": global_incident_lights,
        #"incident_visibility": incident_visibility,
        "diffuse_light":  diffuse_light, #specular
        "specular": specular,
        "incident_sun_light": sun_light ,
    }

    return pbr, extra_results

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

