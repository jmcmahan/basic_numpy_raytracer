#!/usr/bin/env python3

import pygame
import numpy as np

class Toolbox:
    def __init__(self, canvas_size=(512,512), frame_rate=30.0, update_func=lambda tics: None):
        pygame.init()
        self.canvas = pygame.display.set_mode(canvas_size)
        self.update_func = update_func
        self.frame_rate = frame_rate


    def set_title(self, title):
        pygame.display.set_caption(title)

    def blit(self, pixels):
        """Input pixels is MxNx3 array of floats 0 to 1"""

        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            surf = pygame.surfarray.make_surface((255.0*pixels).astype('uint8'))
            self.canvas.blit(surf, (0,0))
            pygame.display.update()

        pygame.quit()


    def blit_from_update(self):
        running = True
        t0 = pygame.time.get_ticks()
        clock = pygame.time.Clock()
        t_total = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            t1 = pygame.time.get_ticks()
            dt = t1 - t0
            t_total = t1
            pixels = self.update_func(dt, t_total)
            if type(pixels) == np.ndarray:
                surf = pygame.surfarray.make_surface((255.0*pixels).astype('uint8'))
                self.canvas.blit(surf, (0,0))
                t0 = pygame.time.get_ticks()
                clock.tick(self.frame_rate)
                pygame.display.update()


        pygame.quit()

    def __del__(self):
        pygame.quit()

