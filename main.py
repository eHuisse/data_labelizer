import pygame
from pygame.locals import *
import cv2
import numpy as np
import sys
import oscilloscope as osc
import screen as sc
import data_parser as dp
import label_draw as ld
import csv
import tkinter as tk
from tkinter import filedialog

def rec_csv(complete_time, complete_field_left, complete_field_right, complete_label, path):
    path = path.replace('.hdf5', '')
    len_time = len(complete_time)
    len_right = len(complete_field_right)
    len_left = len(complete_field_left)
    len_label = len(complete_label)
    if not len_time == len_right == len_left == len_label:
        print(" Weird : lentime = " + str(len_time) + " ; lenleft = " + str(len_left) + " ; lenright = " + str(len_right) + " ; lenlabel = " + str(len_label))
        raise IndexError

    complete_time = complete_time - complete_time[0]
    with open(path + '.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['time', 'left_field', 'right_field', 'label;S0F1R2L3'])
        for i in range(len_time):
            spamwriter.writerow([complete_time[i], complete_field_left[i], complete_field_right[i], complete_label[i]])

        sys.exit(0)


def ask_for_ending():
    non_end = True
    while non_end:
        for event in pygame.event.get():
            if event.key == K_ESCAPE:
                return False
            if event.key == K_RETURN:
                return True

def main():
    win = tk.Tk()
    win.filename = filedialog.askdirectory(initialdir="~", title="Select file")
    print(dp.path_finder(win.filename))

    pygame.init()
    pygame.display.set_caption("video display")
    screen = pygame.display.set_mode([1700, 856])
    imageDisplay = sc.PGVideoDisplay(screen, size=(900, 450), position=(0, 0))
    oscilloDisplay = osc.PGOscilloscope(screen, size=(1700, 400), position=(0, 456), max_value=32768)
    labelDisplay = ld.PGlabeldraw(screen, size=(1700, 0), position=(0, 450))
    video_path, h5_path = dp.path_finder(win.filename) #        dp.path_finder(win.filename)
    count, list_of_frame = dp.video_to_frames(video_path)
    dset = dp.Data_set(h5_path)
    index = 0

    def actualize_frame(index):
        imageDisplay.update(list_of_frame[index])
        print(dset.allDset['image_time_dataset'][index])
        tinf = dset.allDset['image_time_dataset'][index]
        tsup = dset.allDset['image_time_dataset'][index + 1]
        left = dset.get_index_range_from_time((tinf, tsup), 'field_left_dataset')
        right = dset.get_index_range_from_time((tinf, tsup), 'field_right_dataset')
        print(tinf, tsup)
        labelDisplay.update(dset.get_index_range_from_time((tinf, tsup), 'complete_direction'))
        oscilloDisplay.update(left, right)

    while True:
        try:
            #imageDisplay.update(list_of_frame[index])
            #index = index + 1
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == K_RIGHT:
                        index = index + 1
                        try:
                            actualize_frame(index)
                        except IndexError:
                            print('Out of bound')
                            index = index - 1
                            actualize_frame(index)

                    if event.key == K_LEFT:
                        index = index - 1
                        try:
                            actualize_frame(index)
                        except IndexError:
                            print('Out of bound')
                            index = index + 1
                            actualize_frame(index)

                    if event.key == K_DOWN:
                        tinf = dset.allDset['image_time_dataset'][index]
                        tsup = dset.allDset['image_time_dataset'][index + 1]
                        lists = dset.get_index_range_from_time((tinf, tsup))
                        dset.labelize_directions(lists, dset.STOP)
                        labelDisplay.update(dset.get_index_range_from_time((tinf, tsup), 'complete_direction'))

                    if event.key == K_UP:
                        tinf = dset.allDset['image_time_dataset'][index]
                        tsup = dset.allDset['image_time_dataset'][index + 1]
                        lists = dset.get_index_range_from_time((tinf, tsup))
                        dset.labelize_directions(lists, dset.FORWARD)
                        labelDisplay.update(dset.get_index_range_from_time((tinf, tsup), 'complete_direction'))

                    if event.key == K_RETURN:
                        answer = ask_for_ending()
                        if answer:
                            print('Redording')
                            rec_csv(dset.allDset['complete_time_field'],
                                    dset.allDset['field_left_dataset'],
                                    dset.allDset['field_right_dataset'],
                                    dset.allDset['complete_direction'], h5_path)


        except KeyboardInterrupt:
            pygame.quit()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
