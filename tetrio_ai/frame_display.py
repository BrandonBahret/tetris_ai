from threading import Thread

from FrameGrabber.Frame import Frame
import pygame
import numpy as np
import time

import queue

cmap = {
    0 : (97, 18, 117),
    1 : (214, 214, 43),
    2 : (59, 189, 8),
    3 : (201, 57, 38),
    4 : (38, 201, 185),
    5 : (60, 60, 255),
}

def map_colors(blocks, color_map):
    img = blocks.astype(np.int64).copy()
    img = Frame.change_nchannels(img, 3)
    mapped_img = img.copy()
    for key, color in color_map.items():
        mapped_img[mapped_img[:,:,0] == key] = color

    return mapped_img

class FrameDisplay:
    def __init__(self, frame):
        pygame.display.init()
        self.running = True
        self.events = queue.LifoQueue()
        Thread(target=self.__show_image, args=(frame,)).start()
        # Process(target=self.__show_image, args=(frame,)).start()

    def update(self, frame):
        self.frame = frame.copy()
        self.frame = np.fliplr(self.frame)
        self.frame = np.rot90(self.frame, 1)
        self.surf = pygame.surfarray.make_surface(self.frame)
        self.width = self.surf.get_width()
        self.height = self.surf.get_height()

    def get_events(self):
        events = []
        while not self.events.empty():
            events.append(self.events.get(block=False))
        
        return events

    def __show_image(self, frame):
        self.update(frame)
        w, h = self.width, self.height
        self.display = pygame.display.set_mode((w, h))
        self.events = queue.LifoQueue()

        while self.running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.running = False
                if e.type == pygame.MOUSEBUTTONDOWN:
                    self.events.put(e)

            self.display.blit(self.surf, (0, 0))
            pygame.display.update()
            time.sleep(0.01667)
        
        pygame.display.quit()

    def close(self):
        self.running = False

if __name__ == "__main__":
    img = np.zeros((23, 10))
    img[10:] = 1
    img = map_colors(img, cmap)
    img = Frame.resize(img, 300)
    disp = FrameDisplay(img)
