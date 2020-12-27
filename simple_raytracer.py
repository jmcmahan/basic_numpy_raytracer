#!/usr/bin/env python3

import numpy as np
from PIL import Image



def interp2_img_square(img, x, y, oob_color=(0.0,0.0,0.0)):
    """Interpolate on a 2D square image with coordinates 0 to 1.
       An optional color to use when out-of-bounds can be provided as oob_color"""
    img_out = np.tile(np.array(oob_color).reshape((1,3)), (x.shape[0], 1))
    ind_in_bounds = np.where(   (0.0 <= x) 
                              & (x <= 1.0) 
                              & (0.0 <= y) 
                              & (y <= 1.0))
    if len(ind_in_bounds[0]) == 0:
        return img_out

    nx, ny, _ = img.shape

    ix = x[ind_in_bounds] * (nx - 1)
    ix_0 = np.floor(ix).astype('int')
    dx = ix - ix_0
    ix_1 = ix_0 + 1
    ix_1[ix_1 == nx] = nx - 1

    iy = y[ind_in_bounds] * (ny - 1)
    iy_0 = np.floor(iy).astype('int')
    dy = iy - iy_0
    iy_1 = iy_0 + 1
    iy_1[iy_1 == ny] = ny - 1

    if len(dx.shape) == 1:
        dx = dx.reshape((dx.shape[0], 1))
    if len(dy.shape) == 1:
        dy = dy.reshape((dy.shape[0], 1))

    # This is 2D linear interpolation
    img_out[ind_in_bounds, :] =   img[ix_0, iy_0, :] * (1.0 - dx) * (1.0 - dy) \
                                + img[ix_1, iy_0, :] * (      dx) * (1.0 - dy) \
                                + img[ix_0, iy_1, :] * (1.0 - dx) * (      dy) \
                                + img[ix_1, iy_1, :] * (      dx) * (      dy)
    return img_out


def dot_array(x, y):
    """Dot product of array along last axis"""
    p = x * y
    return np.sum(p, axis=len(p.shape) - 1)


def reflect_array(incident, normal):
    return incident - 2.0 * normal \
            * dot_array(incident, normal).reshape((normal.shape[0],1))

class Light:
    def __init__(self, position=(-20.0,20.0,20.0), intensity=1.0):
        self.position = np.array(position)
        self.intensity = intensity

class Material:
    def __init__(self, color=(0.5,0.5,0.5), texture=[],
                              rcoeff_diffuse=1.0,
                              rcoeff_specular=0.2,
                              rcoeff_reflect=0.0,
                              exp_specular=10.0):
        self.color = np.array(color)
        self.texture = texture
        self.rcoeff_diffuse = rcoeff_diffuse
        self.rcoeff_specular = rcoeff_specular
        self.rcoeff_reflect  = rcoeff_reflect
        self.exp_specular = exp_specular


    def get_color(self, xy):
        out_color = np.tile(self.color.reshape(1,3), (xy.shape[0], 1))
        # Only need to do something if the texture is a numpy array
        # representing an image. Otherwise, we're just returning the
        # background color
        if type(self.texture) == np.ndarray:
           nx = self.texture.shape[0] 
           ny = self.texture.shape[1] 



class Sphere:
    def __init__(self, center=(0.0, 0.0, -10.0), radius=1.0, 
                       scale_texture=1.0, yaw=0.0, pitch=0.0, roll=0.0, 
                       material=Material()):
        self.center = np.array(center)
        self.radius = radius
        self.material = material
        self.scale_texture = scale_texture
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll


    def normal(self, intersect_point):
        """Return the normal at a given intersect point. There is no check 
           that the intersection point is valid, but if it's not on the sphere,
           the vector will fail to have length 1.0"""

        return (intersect_point - self.center) / self.radius

    def surface_coords(self, intersect_point):
        """Return spherical coordinates for the surface. These are scaled so the
        'phi' goes from [-1,1] and 'theta' goes from [0,1]. The intersect_point
        variable is an Mx3 array of x,y,z coordinates."""

    

        # These additions make the default orientation of the center of the
        # texture pointing in the positive z axis
        yaw = self.yaw
        pitch = self.pitch + 0.0*90.0
        roll = self.roll + 0.0*90.0

        ca = np.cos(np.deg2rad(yaw))
        sa = np.sin(np.deg2rad(yaw))
        cb = np.cos(np.deg2rad(pitch))
        sb = np.sin(np.deg2rad(pitch))
        cg = np.cos(np.deg2rad(roll))
        sg = np.sin(np.deg2rad(roll))

        # Rotation matrix written to multiply by rows
        A = np.array([ [           ca*cb,            sa*cb,   -sb],
                       [ca*sb*sg - sa*cg, sa*sb*sg + ca*cg, cb*sg],
                       [ca*sb*cg + sa*sg, sa*sb*cg - ca*sg, cb*cg] ])

        p = np.dot(intersect_point - self.center, A)

        # Standard coordinates here
        """
        phi = np.arctan2(p[:,1], p[:,0])

        # This first part calculates x**2 + y**2 for use in the next line
        ss = dot_array(p[:,0:2], p[:,0:2])
        theta = np.arctan2(np.sqrt(ss), p[:,2])
        """

        # Shifted so the texture faces +z by default
        phi = np.arctan2(p[:,2], p[:,1])

        # This first part calculates x**2 + y**2 for use in the next line
        ss = dot_array(p[:,1:], p[:,1:])
        theta = np.arctan2(np.sqrt(ss), p[:,0])

        
        # Negative on the phi coordinates keeps from flipping the image horizontally.
        # Subtracting 0.5 ensures scaling goes from the center of the texture using
        # the convention we've defined where it's 0 to 1. Adding 0.5 back shifts the
        # origin back to its original location

        scale_texture = self.scale_texture
        y =  (phi / np.pi - 0.5) * scale_texture + 0.5
        x = -(theta / np.pi - 0.5) * scale_texture + 0.5

        return(x, y)
    
    def get_color(self, intersect_point):
        """Given an Mx3 array intersect_point, return the color associated with each. 
           Assumes the intersect_points are valid"""
        
        if type(self.material.texture) == list:
            return np.tile(self.material.color, (intersect_point.shape[0], 1))

        elif type(self.material.texture) == np.ndarray:
            u, v = self.surface_coords(intersect_point)
            return interp2_img_square(self.material.texture,
                                        u, v,
                                        oob_color=self.material.color)

        else:
            print("Error: Material texture should be an empty list or a numpy array")

    def intersect_ray(self, origin, rays):
        """Determine rays that intersect the sphere, the intersection points, and normals.
           Returns indices where intersection occurs, distance at those indices and
           points those indices occur at"""
        oc = origin - self.center

        a = dot_array(rays, rays)
        b = 2.0 * dot_array(oc, rays)
        c = dot_array(oc, oc) - self.radius**2

        discriminant = b**2 - 4*a*c
        discriminant[np.where(discriminant < 0.0)] = np.inf

        t1 = (-b + np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2.0 * a)


        t = np.where(np.abs(t1) < np.abs(t2), t1, t2)
        # Have to also rule out where t < 0, so we'll just set it to inf
        ind_intersect = np.where(np.isfinite(t) & (t>=0.0))[0]
        t = t[ind_intersect].reshape((len(ind_intersect), 1))

        # "origin" can be either a single value or an array, so 
        # there's this unfortunate special case
        if len(origin.shape) > 1:
            p_intersect = origin[ind_intersect,:] + t * rays[ind_intersect,:]
        else:
            p_intersect = origin + t * rays[ind_intersect,:]
        
        return (ind_intersect, t, p_intersect)


def scene_intersect(origin, rays, object_list):
    """Run through the object list to find the nearest intersections with the
    given rays. Returns 
    object_ind: index to the nearest intersection object. -1 means none,
                otherwise it's an index into object_list
    p: points in space where the intersection occurs (all inf if none)
    t: distance to nearest interesection (inf if none)
    """

    number_rays = rays.shape[0]

    # This is the index into the object list that indicates which object
    # is hit for this pixel. -1 means no intersection.
    object_ind = np.zeros((number_rays, 1)).astype('int')
    object_ind[:] = -1

    # This is the distance to the intersection
    t = np.zeros((number_rays, 1))
    t[:] = np.inf

    # This is the intersection point
    p = np.zeros((number_rays, 3))
    p[:] = np.inf

    cur_object_ind = 0
    for obj in object_list:
        (intersect_inds, cur_t, cur_p) = obj.intersect_ray(origin, rays)

        update_inds = np.asarray(cur_t < t[intersect_inds]).nonzero()[0]

        object_ind[intersect_inds[update_inds]] = cur_object_ind
        t[intersect_inds[update_inds]] = cur_t[update_inds]
        p[intersect_inds[update_inds], :] = cur_p[update_inds, :]

        cur_object_ind += 1

    return (object_ind, p, t)



# These "get_X_from_object_ind" functions all repeat the index
# calculation. These could be combined, if that would be helpful
def get_color_from_object_ind(object_ind, p, object_list):
    number_rays = object_ind.shape[0]
    colors = np.zeros((number_rays, 3))
    cur_ind = 0
    for obj in object_list:
        ind = np.asarray(object_ind == cur_ind).nonzero()
        colors[ind[0],:] = obj.get_color(p[ind[0],:])
        cur_ind += 1

    return colors


def get_normal_from_object_ind(object_ind, p, object_list):
    number_rays = object_ind.shape[0]
    normals = np.zeros((number_rays, 3))
    cur_ind = 0
    for obj in object_list:
        ind = np.asarray(object_ind == cur_ind).nonzero()
        normals[ind[0],:] = obj.normal(p[ind[0],:])
        cur_ind += 1

    return normals


def get_lighting_parameters_from_object_ind(object_ind, object_list):
    number_rays = object_ind.shape[0]
    rcoeff_diffuse = np.zeros((number_rays, 1))
    rcoeff_specular = np.zeros((number_rays, 1))
    rcoeff_reflect = np.zeros((number_rays, 1))
    exp_specular = np.zeros((number_rays, 1))
    cur_ind = 0
    for obj in object_list:
        ind = np.asarray(object_ind == cur_ind).nonzero()
        rcoeff_diffuse[ind[0],:] = obj.material.rcoeff_diffuse
        rcoeff_specular[ind[0],:] = obj.material.rcoeff_specular
        rcoeff_reflect[ind[0],:] = obj.material.rcoeff_reflect
        exp_specular[ind[0],:] = obj.material.exp_specular
        cur_ind += 1
    return (rcoeff_diffuse, rcoeff_specular, rcoeff_reflect, exp_specular)


def get_lighting(rays, object_ind, p, t, object_list, light_list):
    normals = get_normal_from_object_ind(object_ind, p, object_list)
    rcoeff_diffuse, rcoeff_specular, rcoeff_reflect, exp_specular = \
            get_lighting_parameters_from_object_ind(object_ind, object_list)

    intensity_diffuse = np.zeros(t.shape)
    intensity_specular = np.zeros(t.shape)

    for light in light_list:
        light_ray = light.position - p
        light_distance = np.sqrt(dot_array(light_ray, light_ray))
        light_distance = light_distance.reshape((light_distance.shape[0],1))
        light_ray = light_ray / light_distance


        # The 1e-3 is a small perturbation to avoid automatically intersecting
        # with itself, following the original tutorial suggestion. The dot product
        # sign gives you info on how the incident alight orients relative to the
        # normal, so you add or subtract accordingly

        incidence_diffuse = dot_array(light_ray, normals)
        shadow_origin = p - 1e-3 * normals
        ind = np.where(incidence_diffuse > 0.0)[0]
        shadow_origin[ind, :] = p[ind,:] + 1e-3 * normals[ind,:]

        shadow_object_ind, _, t_light = scene_intersect(shadow_origin, light_ray, object_list)

        
        # Only need to add where there either was no intersection, or the
        # intersection was further than the light. Since I use t = inf
        # when there is no intersection, we can just check the distance
        # is further

        ind_lit = np.asarray(t_light > light_distance).nonzero()[0]

        # Diffuse component

        diffuse_inc = incidence_diffuse[ind_lit] * light.intensity
        diffuse_inc = diffuse_inc.reshape((diffuse_inc.shape[0],1))
        ind = np.asarray(diffuse_inc < 0.0).nonzero()[0]
        diffuse_inc[ind] = 0.0
        
        #intensity_diffuse[ind_lit] = intensity_diffuse[ind_lit] +  diffuse_inc
        intensity_diffuse[ind_lit] += diffuse_inc


        # Specular component

        reflect_ray = reflect_array(light_ray[ind_lit,:], normals[ind_lit,:])
        specular_inc = dot_array(reflect_ray, rays[ind_lit,:])
        specular_inc = specular_inc.reshape((specular_inc.shape[0],1))
        ind = np.asarray(specular_inc < 0.0).nonzero()[0]
        specular_inc[ind] = 0.0
        specular_inc = specular_inc**exp_specular[ind_lit] * light.intensity
        intensity_specular[ind_lit] += specular_inc
 #       intensity_specular[ind_lit] += intensity_specular[ind_lit] + specular_inc



    intensity_diffuse = intensity_diffuse * rcoeff_diffuse
    intensity_specular = intensity_specular * rcoeff_specular
    return (intensity_diffuse, intensity_specular)




def cast_rays(origin, rays, object_list, light_list, depth=0):
    max_reflections = 4
    number_rays = rays.shape[0]
    img = np.zeros((number_rays, 3))
    # Later - consider adding something better for the background, like a 
    # skybox, spherical texture, or at least a way to pass the background
    # info
    #bg_color = np.array((171.0/255.0, 218.0/255.0, 221.0/255.0))
    bg_color = 0.3*np.array((0.5*171.0/255.0, 0.7*218.0/255.0, 221.0/255.0))
    bg_color = 0.8*np.array((0.1,0.11,0.2))



    # Don't need to do anything if we've exceeded the max recursion
    # except return the background
    if depth > max_reflections:
        img[:,:] = bg_color 
        return img

    object_ind, p, t = scene_intersect(origin, rays, object_list)



    # Fill in background anywhere no intersection occurred
    ind_nohit = np.where(object_ind == -1)
    img[ind_nohit,:] = bg_color


    # Now we can handle where rays hit an object 

    ind_hit = np.where(object_ind != -1)[0]
    rays = rays[ind_hit,:]
    object_ind = object_ind[ind_hit]
    p = p[ind_hit,:] 
    t = t[ind_hit]



    """
    intensity_diffuse, intensity_specular = get_lighting(
                                        rays, object_ind, 
                                        p, t, 
                                        object_list, light_list)
    """
    # Getting parameters and other relevant info
    normals = get_normal_from_object_ind(object_ind, p, object_list)
    reflect_rays = reflect_array(rays, normals)
    reflect_length = np.sqrt(dot_array(reflect_rays, reflect_rays))
    reflect_length = reflect_length.reshape((reflect_length.shape[0],1))
    reflect_rays = reflect_rays / reflect_length

    reflect_orient = dot_array(reflect_rays, normals)
    reflect_origin = p - 1e-3 * normals
    ind = np.where(reflect_orient > 0.0)[0]
    reflect_origin[ind, :] = p[ind,:] + 1e-3 * normals[ind, :]
    reflect_color = cast_rays(reflect_origin, reflect_rays, 
                             object_list, light_list, depth+1)


    rcoeff_diffuse, rcoeff_specular, rcoeff_reflect, exp_specular = \
            get_lighting_parameters_from_object_ind(object_ind, object_list)

    intensity_diffuse = np.zeros(t.shape)
    intensity_specular = np.zeros(t.shape)


    # Handle reflections

    # These are the material colors. 
    colors = get_color_from_object_ind(object_ind, p, object_list)


    for light in light_list:
        light_ray = light.position - p
        light_distance = np.sqrt(dot_array(light_ray, light_ray))
        light_distance = light_distance.reshape((light_distance.shape[0],1))
        light_ray = light_ray / light_distance


        # The 1e-3 is a small perturbation to avoid automatically intersecting
        # with itself, following the original tutorial suggestion. The dot product
        # sign gives you info on how the incident alight orients relative to the
        # normal, so you add or subtract accordingly

        incidence_diffuse = dot_array(light_ray, normals)
        shadow_origin = p - 1e-3 * normals
        ind = np.where(incidence_diffuse > 0.0)[0]
        shadow_origin[ind, :] = p[ind,:] + 1e-3 * normals[ind,:]

        shadow_object_ind, _, t_light = scene_intersect(shadow_origin, light_ray, object_list)

        
        # Only need to add where there either was no intersection, or the
        # intersection was further than the light. Since I use t = inf
        # when there is no intersection, we can just check the distance
        # is further

        ind_lit = np.asarray(t_light > light_distance).nonzero()[0]

        # Diffuse component

        diffuse_inc = incidence_diffuse[ind_lit] * light.intensity
        diffuse_inc = diffuse_inc.reshape((diffuse_inc.shape[0],1))
        ind = np.asarray(diffuse_inc < 0.0).nonzero()[0]
        diffuse_inc[ind] = 0.0
        
        #intensity_diffuse[ind_lit] = intensity_diffuse[ind_lit] +  diffuse_inc
        intensity_diffuse[ind_lit] += diffuse_inc


        # Specular component

        reflect_ray = reflect_array(light_ray[ind_lit,:], normals[ind_lit,:])
        specular_inc = dot_array(reflect_ray, rays[ind_lit,:])
        specular_inc = specular_inc.reshape((specular_inc.shape[0],1))
        ind = np.asarray(specular_inc < 0.0).nonzero()[0]
        specular_inc[ind] = 0.0
        specular_inc = specular_inc**exp_specular[ind_lit] * light.intensity
        intensity_specular[ind_lit] += specular_inc
 #       intensity_specular[ind_lit] += intensity_specular[ind_lit] + specular_inc



    intensity_diffuse = intensity_diffuse * rcoeff_diffuse
    intensity_specular = intensity_specular * rcoeff_specular

    img[ind_hit,:] =                 colors * intensity_diffuse \
                     + np.array((1.,1.,1.)) * intensity_specular\
                     + rcoeff_reflect * reflect_color


    # Some adjustments to handle where colors saturate.
    color_max = np.max(img, axis=1)

    ind = np.asarray(color_max > 1.0).nonzero()[0]
    img[ind, :] = img[ind, :] / color_max[ind].reshape((ind.shape[0],1))

    return img



def render(object_list, light_list, canvas_size=(256,256), origin=(0.0,0.0,0.0),
          fov_degrees=90.0, transform=np.eye(3)):
    width = canvas_size[0]
    height = canvas_size[1]
    aspect_ratio = width / height
    length_y = 2.0*np.tan(np.deg2rad(fov_degrees * 0.5))
    length_x = aspect_ratio * length_y

    dx = length_x / width
    dy = length_y / height
    x =  (np.arange( width) + 0.5 - 0.5 *  width) * dx
    y = -(np.arange(height) + 0.5 - 0.5 * height) * dy
    z = np.array(-1.0)


    x,y,z = np.meshgrid(x, y, z, indexing='ij')

    rays = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    length_rays = np.sqrt(dot_array(rays, rays))
    length_rays = length_rays.reshape((length_rays.shape[0],1))
    rays = rays / length_rays

    rays = np.matmul(rays, transform)

    img = cast_rays(origin, rays, object_list, light_list, depth=0)

    img = img.reshape((width, height, 3))
    return img
