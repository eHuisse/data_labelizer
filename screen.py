import pygame
from pygame.locals import *
import cv2
import numpy as np
import sys

def video_to_frames(vidcap):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    count = 0
    list_of_frame = []

    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            list_of_frame.append(image)
            count += 1
        else:
            break
    return count, list_of_frame

class PGVideoDisplay():
    def __init__(self, display_surf, size=(100, 100), position=(0, 0)):
        self.display_surf = display_surf
        self.size = size
        self.position = position
        self.BLACK = (0, 0, 0)

    def update(self, image):
        pygame.draw.rect(self.display_surf, self.BLACK, [self.position[0], self.position[1], self.size[0], self.size[1]])
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.rot90(image)
        image = pygame.surfarray.make_surface(image)
        self.display_surf.blit(image, (self.position[0], self.position[1]))
        pygame.display.update()

if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption("video display")
    screen = pygame.display.set_mode([1280, 720])
    imageDisplay = PGVideoDisplay(screen, size=(1280, 720), position=(0, 0))
    video = cv2.VideoCapture('Test11/Test11.avi')
    count, list_of_frame = video_to_frames(video)
    index = 0
    print(count)

    while True:
        try:
            #imageDisplay.update(list_of_frame[index])
            #index = index + 1
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == K_RIGHT:
                        index = index + 1
                        imageDisplay.update(list_of_frame[index])

                    if event.key == K_LEFT:
                        index = index - 1
                        imageDisplay.update(list_of_frame[index])

        except KeyboardInterrupt:
            pygame.quit()
            cv2.destroyAllWindows()
