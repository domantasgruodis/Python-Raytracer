def patch_scene_class():
    from sightpy.scene import Scene
    
    original_render = Scene.render
    
    def patched_enable_quantum_raytracing(self, config=None):
        self.use_quantum_raytracing = True
        
        from ..quantum import QTraceConfig
        if config is None:
            self.quantum_config = QTraceConfig(
                use_image_coherence=True,
                use_termination_criterion=True,
                max_iterations=3,
                shots_per_search=1024,
                debug=False,
                use_fallback=True,
                primitive_min_check=3
            )
        else:
            self.quantum_config = config
            
        print("Enhanced quantum ray tracing enabled")
        
        if self.quantum_config.use_image_coherence and hasattr(self, 'camera'):
            from ..quantum.quantum_raytracer import QuantumNeighborData
            self.quantum_neighbor_data = QuantumNeighborData(
                self.camera.screen_width, 
                self.camera.screen_height
            )
    
    def patched_quantum_get_raycolor(ray, scene):
        from ..utils.vector3 import rgb
        from ..ray import get_raycolor, Ray
        import numpy as np
        
        total_pixels = len(ray.origin.x)
        
        result_color = rgb(
            np.zeros(total_pixels),
            np.zeros(total_pixels),
            np.zeros(total_pixels)
        )
        
        from ..quantum import trace_ray
        
        batch_size = 100
        
        for batch_start in range(0, total_pixels, batch_size):
            batch_end = min(batch_start + batch_size, total_pixels)
            batch_size_actual = batch_end - batch_start
            
            for i in range(batch_start, batch_end):
                pixel_mask = np.zeros(total_pixels, dtype=bool)
                pixel_mask[i] = True
                
                y = i // scene.camera.screen_width
                x = i % scene.camera.screen_width
                
                try:
                    single_ray = ray.extract(pixel_mask)
                    
                    single_ray.pixel_x = x
                    single_ray.pixel_y = y
                    
                    primitive_idx, distance, normal = trace_ray(single_ray, scene, scene.quantum_config)
                    
                    if primitive_idx < 0 and scene.quantum_config.use_fallback:
                        if np.random.random() < 0.1:
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
                            
                            result_color.x[i] = pixel_color.x
                            result_color.y[i] = pixel_color.y
                            result_color.z[i] = pixel_color.z
                            
                            continue
                    
                    if primitive_idx >= 0:
                        primitive = scene.scene_primitives[primitive_idx]
                        material = primitive.material
                        collider = primitive.collider_list[0]
                        
                        hit_point = single_ray.origin + single_ray.dir * distance
                        
                        hit = type('Hit', (), {
                            'collider': collider,
                            'distance': distance,
                            'point': hit_point,
                            'surface': primitive,
                            'material': material,
                            'orientation': UPWARDS if normal.dot(single_ray.dir) < 0 else UPDOWN,
                            'N': normal,
                            'get_uv': lambda: primitive.get_uv(type('TempHit', (), {
                                'point': hit_point, 
                                'collider': collider
                            }))
                        })
                        
                        pixel_color = material.get_color(scene, single_ray, hit)
                        
                        result_color.x[i] = pixel_color.x
                        result_color.y[i] = pixel_color.y
                        result_color.z[i] = pixel_color.z
                    
                except Exception as e:
                    try:
                        classical_ray = Ray(
                            origin=ray.origin.extract(pixel_mask),
                            dir=ray.dir.extract(pixel_mask),
                            depth=ray.depth,
                            n=ray.n.extract(pixel_mask),
                            reflections=ray.reflections,
                            transmissions=ray.transmissions,
                            diffuse_reflections=ray.diffuse_reflections
                        )
                        pixel_color = get_raycolor(classical_ray, scene)
                        
                        result_color.x[i] = pixel_color.x
                        result_color.y[i] = pixel_color.y
                        result_color.z[i] = pixel_color.z
                    except:
                        pass
        
        return result_color
    
    def patched_render(self, samples_per_pixel, progress_bar=False):
        if self.use_quantum_raytracing and hasattr(self, 'quantum_config'):
            from ..ray import get_raycolor
            
            original_quantum_get_raycolor = patched_quantum_get_raycolor
            
            try:
                self._quantum_get_raycolor = patched_quantum_get_raycolor
                
                return original_render(self, samples_per_pixel, progress_bar)
            finally:
                self._quantum_get_raycolor = None
        else:
            return original_render(self, samples_per_pixel, progress_bar)

    Scene.enable_quantum_raytracing = patched_enable_quantum_raytracing
    Scene.render = patched_render
    
    from ..utils.constants import FARAWAY, UPWARDS, UPDOWN
    
    print("Scene class successfully patched with enhanced quantum ray tracing")
    
    return True