#!/usr/bin/env python3

import numpy as np
from importlib import reload
import simple_raytracer 
reload(simple_raytracer)
from simple_raytracer import *

from graphics_tools import Toolbox
from PIL import Image
import time



run_type = 'single_view'
run_type = 'updated'
run_type = 'debug'
run_type = 'make_images'

def load_tex(filename):
    tex = np.asarray(Image.open(filename)).astype('float') / 255
    tex = tex.transpose((1,0,2)).copy(order='C')
    return tex

tex_kid = np.asarray(Image.open('kid_head.png')).astype('float') / 255
tex_kid = tex_kid.transpose((1,0,2)).copy(order='C')

tex_cat = np.asarray(Image.open('cat_head.png')).astype('float') / 255
tex_cat = tex_cat.transpose((1,0,2)).copy(order='C')

tex_dog = np.asarray(Image.open('dog_head.png')).astype('float') / 255
tex_dog = tex_dog.transpose((1,0,2)).copy(order='C')

tex_tree = np.asarray(Image.open('snow.png')).astype('float') / 255
tex_tree = tex_tree.transpose((1,0,2)).copy(order='C')


tex_man_happy = load_tex('man_happy.png')
tex_man_angry = load_tex('man_angry.png')

tex_woman_happy = load_tex('woman_happy.png')
tex_woman_angry = load_tex('woman_angry.png')

tex_merry_xmas = load_tex('merry_xmas.png')

material_kid = Material(color=(91.0/255.0, 78.0/255.0, 236.0/255.0), 
                            texture=tex_kid,
                            rcoeff_diffuse=0.5,
                            rcoeff_specular=0.3,
                            rcoeff_reflect=0.1,
                            exp_specular=50.0)


material_cat = Material(color=(211.0/255.0, 48.0/255.0, 57.0/255.0), 
                            texture=tex_cat,
                            rcoeff_diffuse=0.5,
                            rcoeff_specular=0.3,
                            rcoeff_reflect=0.1,
                            exp_specular=50.0)

material_dog = Material(color=(214.0/255.0, 225.0/255.0, 87.0/255.0), 
                            texture=tex_dog,
                            rcoeff_diffuse=0.5,
                            rcoeff_specular=0.3,
                            rcoeff_reflect=0.1,
                            exp_specular=50.0)

material_man_angry = Material(color=(0.0/255.0, 150.0/255.0, 0.0/255.0), 
                            texture=tex_man_angry,
                            rcoeff_diffuse=0.5,
                            rcoeff_specular=0.3,
                            rcoeff_reflect=0.1,
                            exp_specular=50.0)
material_man_happy = Material(color=(0.0/255.0, 150.0/255.0, 0.0/255.0), 
                            texture=tex_man_happy,
                            rcoeff_diffuse=0.5,
                            rcoeff_specular=0.3,
                            rcoeff_reflect=0.1,
                            exp_specular=50.0)

material_woman_happy = Material(color=(200.0/255.0, 0.0/255.0, 0.0/255.0), 
                            texture=tex_woman_happy,
                            rcoeff_diffuse=0.5,
                            rcoeff_specular=0.3,
                            rcoeff_reflect=0.1,
                            exp_specular=50.0)

material_woman_angry = Material(color=(200.0/255.0, 0.0/255.0, 0.0/255.0), 
                            texture=tex_woman_angry,
                            rcoeff_diffuse=0.5,
                            rcoeff_specular=0.3,
                            rcoeff_reflect=0.1,
                            exp_specular=50.0)
material_xmas = Material(color=(201.0/255.0, 225.0/255.0, 226.0/255.0), 
                            texture=tex_merry_xmas,
                            rcoeff_diffuse=0.6,
                            rcoeff_specular=0.4,
                            rcoeff_reflect=0.2,
                            exp_specular=60.0)
"""
material_tree = Material(color=(99.0/255.0, 118.0/255.0, 47.0/255.0), 
                            texture=tex_tree,
                            rcoeff_diffuse=0.5,
                            rcoeff_specular=0.1,
                            rcoeff_reflect=0.0,
                            exp_specular=40.0)
"""

material_tree = Material(color=(241.0/255.0, 241.0/255.0, 252.0/255.0), 
                            texture=tex_tree,
                            rcoeff_diffuse=0.5,
                            rcoeff_specular=0.1,
                            rcoeff_reflect=0.0,
                            exp_specular=40.0)


material_mirror = Material(color=(1.0,1.0,1.0),
                            rcoeff_diffuse=0.0,
                            rcoeff_specular=10.0,
                            rcoeff_reflect=0.7,
                            exp_specular=1425.0)

s_kid = Sphere(center=(0.0, 0.0, -4.0), radius=1.0, 
                    material=material_kid, scale_texture=1.9)
s_cat = Sphere(center=(2.5, 0.0, -4.0), radius=1.0, 
                    material=material_cat, scale_texture=2.0)
s_dog = Sphere(center=(-2.5, 0.0, -4.0), radius=1.0,
                    material=material_dog, scale_texture=2.0)

s_tree1 = Sphere(center=(0.0, -5.0, -10.25), radius=6.0,
                    material=material_tree, scale_texture=0.5)
s_tree2 = Sphere(center=(0.0, 0.0, -10.25), radius=4.5,
                    material=material_tree, scale_texture=0.5)
s_tree3 = Sphere(center=(0.0, 4.0, -10.25), radius=3.0,
                    material=material_tree, scale_texture=0.5)

s_mirror = Sphere(center=(0.0,0.0,106.0), radius=100.0,
                    material=material_mirror)

s_man_angry = Sphere(center=(0.0, -7.0, 2.0), radius=3.0,
                    material=material_man_angry, scale_texture=2.0,
                    pitch=180.0)
s_man_happy = Sphere(center=(0.0, -7.0, 2.0), radius=3.0,
                    material=material_man_happy, scale_texture=2.0,
                    pitch=180.0)
s_woman_angry = Sphere(center=(0.0, 7.0, 2.0), radius=3.0,
                    material=material_woman_angry, scale_texture=2.0,
                    pitch=180.0)
s_woman_happy = Sphere(center=(0.0, 7.0, 2.0), radius=3.0,
                    material=material_woman_happy, scale_texture=2.0,
                    pitch=180.0)
s_xmas = Sphere(center=(0.0, 0.0, 2.0), radius=3.0,
                    material=material_xmas, scale_texture=1.5,
                    pitch=180.0)


l1 = Light(position=(-5.0,5.0,5.0), intensity=2.5)
#l1 = Light(position=(-1.0,0.0,1.0), intensity=2.5)
l2 = Light(position=(20.0, 20.0,-20.0), intensity=1.0)
l3 = Light(position=(-20.0, 20.0,-20.0), intensity=1.0)


object_list = [s_kid, s_cat, s_dog, s_tree1, s_tree2, s_tree3, s_mirror,
              s_man_angry, s_woman_happy, s_xmas]
light_list = [l1, l2, l3]


origin = np.array((0.0, 0.0, 0.0))

canvas_width =  256 
canvas_height = 256

if run_type == 'updated':
    canvas_width =  160*2 
    canvas_height = 90 *2
else:
    canvas_width =  320 * 4
    canvas_height = 180 * 4
canvas_size = (canvas_width, canvas_height)


def get_lookat_mat(v_from, v_to, v_up=(0.0,1.0,0.0)):
    v_from = np.array(v_from)
    v_to = np.array(v_to)
    v_up = np.array(v_up)

    v1 = v_from - v_to
    l = np.linalg.norm(v1)
    v_z = v1 / l

    l = np.linalg.norm(v_up)
    v_up = v_up / l

    v_x = np.cross(v_up, v_z)

    v_y = np.cross(v_z, v_x)

    R = np.array([v_x,v_y,v_z])
    return R


z_lineup = -4.0
fov_degrees = 90.0

def update_func(dt, t_total):
    t = t_total / 1000.0

    yaw1 = 20.0*np.sin(np.pi*2.0*t*0.5)
    #yaw2 = 45.0*np.sin(2.5*t) 
    yaw2 = 45.0*np.sin(np.pi*2.0*t*0.5) 
    object_list[0].pitch = yaw1
    object_list[0].yaw = yaw1
    object_list[1].yaw =  yaw2
    object_list[2].yaw = -yaw2
    r = 3.0

    object_list[0].center = (0.0, 
                             abs(np.sin(np.pi*2.0*t))**2, 
                             z_lineup)
    object_list[1].center = (r*np.cos(t*2.0*np.pi/8.0), 
                             r*np.sin(t*2.0*np.pi/8.0), 
                             z_lineup)
    object_list[2].center = (r*np.cos(t*2.0*np.pi/8.0 + np.pi), 
                             r*np.sin(t*2.0*np.pi/8.0 + np.pi), 
                             z_lineup)

    # Update textures
    if np.floor(t).astype(np.int) % 2 == 0:
        object_list[7] = s_man_angry
        object_list[8] = s_woman_happy
    else:
        object_list[7] = s_man_happy
        object_list[8] = s_woman_angry

    # Shift folks around
    object_list[7].center = (7.0*np.sin(t*2.0*np.pi/8.0),-7.0,3.0)
    object_list[8].center = (7.0*np.sin(t*2.0*np.pi/8.0), 7.0,3.0)
    object_list[9].roll = np.sin(t*2.0*np.pi/2.0)*30.0
    object_list[9].center = (7.0*np.sin(t*2.0*np.pi/8.0), 0.0,3.0)


    origin = np.array((12.0*np.cos(2.0*np.pi*t/32.0 - 0.5*np.pi), 
                          np.sin(2.0*np.pi*t/32.0 )*10.0, 
                        -(12.0*np.sin(2.0*np.pi*t/32.0  - 0.5*np.pi) + 13.0)))
    R = get_lookat_mat(origin, (0.0, 0.0, -10.0))
    img = render(object_list, light_list, canvas_size, origin, 
                 fov_degrees=fov_degrees, transform=R)
    return img







if run_type == 'single_view':
    frame_rate = 30.0
    tb = Toolbox(canvas_size, frame_rate, update_func)
    tb.set_title('Christmas')
    start_time = time.time()
    img = render(object_list, light_list, canvas_size, origin, fov_degrees=fov_degrees)
    print(time.time() - start_time)
    tb.blit(img)
elif run_type == 'updated':
    frame_rate = 30.0
    tb = Toolbox(canvas_size, frame_rate, update_func)
    tb.set_title('Christmas')
    tb.blit_from_update()

elif run_type == 'debug':
    frame_rate = 30.0
    tb = Toolbox(canvas_size, frame_rate, update_func)
    tb.set_title('Christmas')
    img = update_func(0, 25000)
    tb.blit(img)

elif run_type == 'make_images':
    frame_rate = 60.0
    tmax = 32.0
    frames = tmax*frame_rate
    t_list = 1000.0 * np.arange(frames) / frame_rate
    dt = 0.0
    it = 1
    start_time = time.time()
    for t in t_list:
        img = update_func(0, t)
        framecur = (255.0*img.transpose(1,0,2)).astype(np.uint8)
        im = Image.fromarray(framecur)
        outname = "raytrace_xmas_plus/image%03d.png" % it
        im.save(outname)
        cur_time = time.time() - start_time
        frac_done = it / frames
        etr = round(cur_time / frac_done * (1-frac_done) / 60.0)
        print('%d of %d, etr: %d minutes' % (it, frames, etr))
        print('Elapsed: %f minutes' % (cur_time/60.0))
        it += 1


