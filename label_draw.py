import pygame
import numpy as np
import time

class PGlabeldraw():
    def __init__(self, display_surf, size=(100, 100), position=(0, 0)):
        self.display_surf = display_surf
        self.size = size
        self.position = position
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.colortable = [self.BLUE, self.YELLOW, self.GREEN, self.RED]


    def update(self, value):
        length = len(value)
        ordo = np.linspace(self.position[0], self.position[0] + self.size[0], length)
        for i in range(length - 1):
            pygame.draw.line(self.display_surf, self.colortable[int(value[i])], [int(ordo[i]), self.position[1]], [int(ordo[i+1]), self.position[1]], 3)

        pygame.display.flip()
