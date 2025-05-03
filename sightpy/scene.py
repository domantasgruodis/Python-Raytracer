from PIL import Image
import numpy as np
import time
from .utils import colour_functions as cf
from .camera import Camera
from .utils.constants import *
from .utils.vector3 import vec3, rgb
from .ray import Ray, get_raycolor, get_distances
from . import lights
from .backgrounds.skybox import SkyBox
from .backgrounds.panorama import Panorama


class Scene():
    def __init__(self, ambient_color = rgb(0.01, 0.01, 0.01), n = vec3(1.0,1.0,1.0), use_quantum_raytracing=False) :
        # n = index of refraction (by default index of refraction of air n = 1.)
        
        self.scene_primitives = []
        self.collider_list = []
        self.shadowed_collider_list = []
        self.Light_list = []
        self.importance_sampled_list = []
        self.ambient_color = ambient_color
        self.n = n
        self.importance_sampled_list = []

        self.use_quantum_raytracing = use_quantum_raytracing
        self.quantum_config = None
        self.quantum_neighbor_data = None

    def enable_quantum_raytracing(self, config=None):
        self.use_quantum_raytracing = True
        
        from .quantum import QTraceConfig
        if config is None:
            self.quantum_config = QTraceConfig()
        else:
            self.quantum_config = config
            
        print("Quantum ray tracing enabled")
        
    def disable_quantum_raytracing(self):
        """Disable quantum ray tracing."""
        self.use_quantum_raytracing = False
        self.quantum_config = None
        print("Quantum ray tracing disabled")


    def add_Camera(self, look_from, look_at, screen_width = 400, screen_height = 300, **kwargs):
        self.camera = Camera(look_from=look_from, look_at=look_at, screen_width=screen_width, screen_height=screen_height, **kwargs)

        if self.use_quantum_raytracing and self.quantum_config and self.quantum_config.use_image_coherence:
            from .quantum.quantum_raytracer import QuantumNeighborData
            self.quantum_neighbor_data = QuantumNeighborData(screen_width, screen_height)


    def add_PointLight(self, pos, color):
        self.Light_list += [lights.PointLight(pos, color)]
        
    def add_DirectionalLight(self, Ldir, color):
        self.Light_list += [lights.DirectionalLight(Ldir.normalize() , color)]  

    def add(self,primitive, importance_sampled = False):
        self.scene_primitives += [primitive]
        self.collider_list += primitive.collider_list

        if importance_sampled == True:
            self.importance_sampled_list += [primitive]

        if primitive.shadow == True:
            self.shadowed_collider_list += primitive.collider_list
            
        
    def add_Background(self, img, light_intensity = 0.0, blur =0.0 , spherical = False):

        primitive = None
        if spherical == False:
            primitive = SkyBox(img, light_intensity = light_intensity, blur = blur)
        else:
            primitive = Panorama(img, light_intensity = light_intensity, blur = blur)

        self.scene_primitives += [primitive]        
        self.collider_list += primitive.collider_list

    def trace_ray_quantum(self, ray, pixel_x=None, pixel_y=None):
        from .quantum import trace_ray

        if pixel_x is not None and pixel_y is not None:
            ray.pixel_x = pixel_x
            ray.pixel_y = pixel_y
        
        primitive_idx, distance, normal = trace_ray(ray, self, self.quantum_config)
        
        if (self.quantum_neighbor_data is not None and
            primitive_idx >= 0 and
            pixel_x is not None and
            pixel_y is not None):
            self.quantum_neighbor_data.update(pixel_x, pixel_y, primitive_idx)
        
        if primitive_idx < 0:
            return rgb(0., 0., 0.)
        
        primitive = self.scene_primitives[primitive_idx]
        material = primitive.material
        
        hit = type('Hit', (), {
            'collider': primitive.collider_list[0],
            'distance': distance,
            'point': ray.origin + ray.dir * distance,
            'surface': primitive,
            'material': material,
            'orientation': UPWARDS if normal.dot(ray.dir) < 0 else UPDOWN,
            'N': normal,
            'get_uv': lambda: primitive.get_uv(type('TempHit', (), {'point': ray.origin + ray.dir * distance, 'collider': primitive.collider_list[0]}))
        })
        
        return material.get_color(self, ray, hit)
        
    def render(self, samples_per_pixel, progress_bar=False):
        print("Rendering...")

        t0 = time.time()
        color_RGBlinear = rgb(0., 0., 0.)

        if progress_bar:
            try:
                import progressbar
                bar_available = True
            except ModuleNotFoundError:
                print("progressbar module is required. \nRun: pip install progressbar")
                bar_available = False
        else:
            bar_available = False

        if bar_available and progress_bar:
            bar = progressbar.ProgressBar()
            sample_range = bar(range(samples_per_pixel))
        else:
            sample_range = range(samples_per_pixel)
            
        using_quantum = self.use_quantum_raytracing and hasattr(self, 'quantum_config')
        if using_quantum:
            print("Using quantum ray tracing")
            
            from .quantum import trace_ray
            
            def quantum_get_raycolor(ray, scene):
                total_pixels = len(ray.origin.x)
                result_color = rgb(
                    np.zeros(total_pixels),
                    np.zeros(total_pixels),
                    np.zeros(total_pixels)
                )
                
                for idx in range(total_pixels):
                    pixel_mask = np.zeros(total_pixels, dtype=bool)
                    pixel_mask[idx] = True
                    
                    y = idx // self.camera.screen_width
                    x = idx % self.camera.screen_width
                    
                    try:
                        single_ray = ray.extract(pixel_mask)
                        
                        single_ray.pixel_x = x
                        single_ray.pixel_y = y
                        
                        primitive_idx, distance, normal = trace_ray(single_ray, scene, scene.quantum_config)
                        
                        if primitive_idx >= 0:
                            classical_ray = Ray(
                                origin=single_ray.origin,
                                dir=single_ray.dir,
                                depth=single_ray.depth,
                                n=single_ray.n,
                                reflections=single_ray.reflections,
                                transmissions=single_ray.transmissions,
                                diffuse_reflections=single_ray.diffuse_reflections
                            )
                            
                            pixel_color = get_raycolor(classical_ray, scene)
                            
                            result_color.x[idx] = pixel_color.x
                            result_color.y[idx] = pixel_color.y
                            result_color.z[idx] = pixel_color.z
                        
                    except Exception as e:
                        print(f"Error processing ray at pixel ({x}, {y}): {e}")
                
                return result_color

        for i in sample_range:
            if using_quantum:
                color_RGBlinear += quantum_get_raycolor(self.camera.get_ray(self.n), self)
            else:
                color_RGBlinear += get_raycolor(self.camera.get_ray(self.n), scene=self)

            if bar_available and progress_bar:
                bar.update(i)

        color_RGBlinear = color_RGBlinear / samples_per_pixel
        
        color = cf.sRGB_linear_to_sRGB(color_RGBlinear.to_array())
        
        print("Render Took", time.time() - t0)

        img_RGB = []
        for c in color:
            img_RGB += [Image.fromarray((255 * np.clip(c, 0, 1).reshape((self.camera.screen_height, self.camera.screen_width))).astype(np.uint8), "L")]

        return Image.merge("RGB", img_RGB)


    def get_distances(self): #Used for debugging ray-primitive collisions. Return a grey map of objects distances.

        print ("Rendering...")
        t0 = time.time()
        color_RGBlinear = get_distances( self.camera.get_ray(self.n), scene = self)
        #gamma correction
        color = color_RGBlinear.to_array()
        
        print ("Render Took", time.time() - t0)


        img_RGB = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((self.camera.screen_height, self.camera.screen_width))).astype(np.uint8), "L") for c in color]
        return Image.merge("RGB", img_RGB)