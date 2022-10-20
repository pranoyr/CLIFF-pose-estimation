from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

import torch

# from nvdiffrast_renderer import MeshRenderer
import numpy as np

    


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    # torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


R, T = look_at_view_transform(2.7, 0, 0) 
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

raster_settings = RasterizationSettings(
    image_size=1000, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)




# Set paths

lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]])


renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

# fov = 2 * np.arctan(112 / 1015.) * 180 / np.pi

# renderer = MeshRenderer(
#             rasterize_fov=fov, znear=5., zfar=15., rasterize_size=int(2 * 112)
#         )
