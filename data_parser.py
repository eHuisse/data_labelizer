import h5py
import numpy as np
import cv2
import sys
import os
from matplotlib import pyplot as plt

def path_finder(argv):
    title = argv.split('/')[-1]
    print(title)
    video_exists = os.path.isfile(str(argv) + '/' + str(title) + '.avi')
    h5_exists = os.path.isfile(str(argv) + '/' + str(title) + '.hdf5')

    if not (video_exists and h5_exists):
        print("video file or signal file is missing")
        sys.exit(0)

    video_path = str(argv) + '/' + str(title) + '.avi'
    h5py_path = str(argv) + '/' + str(title) + '.hdf5'

    return video_path, h5py_path

def video_to_frames(video_path):
    # extract frames from a video and save to directory as 'x.png' where
    # x is the frame index
    count = 0
    list_of_frame = []
    vidcap = cv2.VideoCapture(video_path)

    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            list_of_frame.append(image)
            count += 1
        else:
            break
    return count, list_of_frame[2:-1]

#
# def h5py_parser(h5_path):
#
#     def direction_convolution(dir, kernel_size=5):
#         half_kernel = kernel_size // 2
#         new_dir = np.zeros(len(dir))
#         for i in range(len(dir) - kernel_size):
#             if np.count_nonzero(dir[i:i+kernel_size] == 1) >= 1:
#                 new_dir[i+half_kernel+1] = 1
#         return new_dir.tolist()
#
#     def list_from_index(left_field, right_field, time_field, index_list):
#         wrapped_left_field = []
#         wrapped_right_field = []
#         wrapped_time_field = []
#
#         for i in range(len(index_list)):
#             if len(index_list[i]) == 0:
#                 wrapped_left_field.append([])
#                 wrapped_left_field.append([])
#                 wrapped_left_field.append([])
#             else:
#                 wrapped_left_field.append([left_field[index] for index in index_list[i]])
#                 wrapped_right_field.append([right_field[index] for index in index_list[i]])
#                 wrapped_time_field.append([time_field[index] for index in index_list[i]])
#
#         return wrapped_left_field, wrapped_right_field, wrapped_time_field
#
#     def data_wrapper(time_field, time_frame):
#         index_list = []
#         dropped_frame_index = []
#
#         for i in range(len(time_frame)):
#             try:
#                 t_min = time_frame[i]
#                 t_max = time_frame[i+1]
#             except IndexError:
#                 index_list.append([])
#                 break
#             index_sub_list = []
#
#             j = 0
#             k = 0
#
#             try:
#                 while time_field[j] < t_min:
#                     j = j + 1
#
#                 k = j
#
#                 while time_field[k] < t_max:
#                     index_sub_list.append(k)
#                     k = k + 1
#
#
#                 if k == j:
#                     index_list.append([])
#                     break
#
#                 index_list.append(index_sub_list)
#
#             except IndexError:
#                 index_list.append([])
#
#         #print(index_list, len(time_frame))
#         return index_list
#
#
#     f = h5py.File(h5_path, 'r')
#     flist = list(f)
#     if not len(flist) ==9:
#         print("Their is a Dset missing or too much Dset")
#
#     allDSet = dict()
#
#     for dset in flist:
#         #print(dset)
#         allDSet[dset] = f.get(dset).value
#
#     fieldshape= allDSet['field_right_dataset'][2:-1].shape
#
#     allDSet['direction_left_dataset'] = direction_convolution(np.reshape(allDSet['direction_left_dataset'][2:-1], -1), 5)
#     allDSet['direction_right_dataset'] = direction_convolution(np.reshape(allDSet['direction_right_dataset'][2:-1], -1), 5)
#
#     allDSet['direction_time_dataset'] = np.reshape(allDSet['direction_time_dataset'][2:-1], -1)
#     allDSet['feature_dataset'] = np.reshape(allDSet['feature_dataset'][2:-1], -1)
#     allDSet['feature_time_dataset'] = np.reshape(allDSet['feature_time_dataset'][2:-1], -1)
#     allDSet['field_time_dataset'] = np.reshape(allDSet['field_time_dataset'][2:-1], -1)
#     allDSet['image_time_dataset'] = np.reshape(allDSet['image_time_dataset'][2:-1], -1)
#     allDSet['field_left_dataset'] = np.reshape(allDSet['field_left_dataset'][2:-1], -1)
#     allDSet['field_right_dataset'] = np.reshape(allDSet['field_right_dataset'][2:-1], -1)
#
#     allDSet['complete_time_field'] = np.array([])
#     #print(allDSet['image_time_dataset'].shape)
#
#     diff = allDSet['field_time_dataset'][10] - allDSet['field_time_dataset'][11]
#
#     allDSet['complete_time_field'] = np.linspace(allDSet['field_time_dataset'][0],
#                                                             allDSet['field_time_dataset'][-1] + diff,
#                                                             num=fieldshape[1]*fieldshape[0])
#
#     plt.figure()
#     plt.plot(allDSet['complete_time_field'])
#     plt.show()
#     print(len(allDSet['complete_time_field']), len(allDSet['field_left_dataset']))
#     frame_index_list = data_wrapper(allDSet['complete_time_field'], allDSet['image_time_dataset'])
#     dir_index_list = data_wrapper(allDSet['complete_time_field'], allDSet['direction_time_dataset'])
#     print(frame_index_list[10], len(allDSet['image_time_dataset']))
#     print(dir_index_list, len(allDSet['direction_time_dataset']))
#
#
#
#     return allDSet
#
# def parser(path_name):
#     video_path, h5_path = path_finder(path_name)
#     count, list_of_frame = video_to_frames(video_path)
#     allDset = h5py_parser(h5_path)
#     frame_time_len = len(allDset['image_time_dataset'])
#
#     diff = count - frame_time_len
#
#     list_of_frame = list_of_frame[diff:-1]
#
#     return list_of_frame, allDset

class Data_set():
    def __init__(self, filepath):
        self.filepath = filepath
        self.allDset = self.open_dset(self.filepath)

        self.STOP = 0
        self.FORWARD = 1
        self.RIGHT = 2
        self.LEFT = 3

        self.time_to_index_A = 0
        self.time_to_index_B = 0
        self.index_to_time_A = 0
        self.index_to_time_B = 0

        self.index_to_time_A, \
        self.index_to_time_B, \
        self.time_to_index_A, \
        self.time_to_index_B = self.get_linear_coef_time_index(self.allDset['complete_time_field'])
        self.labelize_directions()


    def direction_convolution(self, dir, kernel_size=5):
        half_kernel = kernel_size // 2
        new_dir = np.zeros(len(dir))
        for i in range(len(dir) - kernel_size):
            if np.count_nonzero(dir[i:i + kernel_size] == 1) >= 1:
                new_dir[i + half_kernel + 1] = 1
        return new_dir.tolist()

    def labelize_directions(self, index_tuple=None, dir=None):
        if index_tuple is not None and dir is not None:
            if type(index_tuple) == tuple:
                index = np.arange(index_tuple[0], index_tuple[1])
                self.allDset['complete_direction'][index] = dir
            else:
                self.allDset['complete_direction'][index_tuple] = dir

        else:
            for i in range(len(self.allDset['direction_right_dataset']) - 1):
                timeinf = self.allDset['direction_time_dataset'][i]
                timesup = self.allDset['direction_time_dataset'][i + 1]

                if self.allDset['direction_right_dataset'][i] == 1:
                    index = self.get_index_range_from_time((timeinf, timesup))
                    self.allDset['complete_direction'][index] = self.RIGHT

                if self.allDset['direction_left_dataset'][i] == 1:
                    index = self.get_index_range_from_time((timeinf, timesup))
                    self.allDset['complete_direction'][index] = self.LEFT

    def open_dset(self, fp):
        f = h5py.File(fp, 'r')
        flist = list(f)
        if not len(flist) == 9:
            print("Their is a Dset missing or too much Dset")

        allDSet = dict()

        for dset in flist:
            # print(dset)
            allDSet[dset] = f[(dset)]

        fieldshape = allDSet['field_right_dataset'][2:-1].shape

        if not fieldshape == allDSet['field_left_dataset'][2:-1].shape:
            print("Different shape of right and left field")
            sys.exit(1)

        allDSet['direction_left_dataset'] = self.direction_convolution(
            np.reshape(allDSet['direction_left_dataset'][2:-1], -1), 5)
        allDSet['direction_right_dataset'] = self.direction_convolution(
            np.reshape(allDSet['direction_right_dataset'][2:-1], -1), 5)

        allDSet['direction_time_dataset'] = np.reshape(allDSet['direction_time_dataset'][2:-1], -1)
        allDSet['feature_dataset'] = np.reshape(allDSet['feature_dataset'][2:-1], -1)
        allDSet['feature_time_dataset'] = np.reshape(allDSet['feature_time_dataset'][2:-1], -1)
        allDSet['field_time_dataset'] = np.reshape(allDSet['field_time_dataset'][2:-1], -1)
        allDSet['image_time_dataset'] = np.reshape(allDSet['image_time_dataset'][2:-1], -1)
        allDSet['field_left_dataset'] = np.reshape(allDSet['field_left_dataset'][2:-1], -1)
        allDSet['field_right_dataset'] = np.reshape(allDSet['field_right_dataset'][2:-1], -1)

        allDSet['complete_time_field'] = np.array([])
        # print(allDSet['image_time_dataset'].shape)

        diff = allDSet['field_time_dataset'][10] - allDSet['field_time_dataset'][11]

        allDSet['complete_time_field'] = np.linspace(allDSet['field_time_dataset'][0],
                                                     allDSet['field_time_dataset'][-1] + diff,
                                                     num=fieldshape[1] * fieldshape[0])
        allDSet['complete_direction'] = np.ones(fieldshape[1] * fieldshape[0])

        return allDSet

    def get_linear_coef_time_index(self, complete_time):
        index_to_time_A = (complete_time[-1] - complete_time[0]) / len(complete_time)
        index_to_time_B = complete_time[0]
        time_to_index_A = 1 / index_to_time_A
        time_to_index_B = -index_to_time_B / index_to_time_A
        return index_to_time_A, index_to_time_B, time_to_index_A, time_to_index_B

    def get_index_from_time(self, time):
        index = int(self.time_to_index_A * time + self.time_to_index_B)
        if index > len(self.allDset['complete_time_field']) or index < 0:
            print('Time out of bound')
            raise IndexError
        return index

    def get_time_from_index(self, index):
        time = self.index_to_time_A * index + self.index_to_time_B
        if time > self.allDset['complete_time_field'][-1] or time < self.allDset['complete_time_field'][0]:
            print('Index out of bound')
            raise IndexError
        return time

    def get_index_range_from_time(self, time_tuple, data=None):
        index_begin = int(self.time_to_index_A * time_tuple[0] + self.time_to_index_B)
        index_end = int(self.time_to_index_A * time_tuple[1] + self.time_to_index_B)
        if index_begin > len(self.allDset['complete_time_field']) or index_begin < 0\
                or index_end > len(self.allDset['complete_time_field']) or index_end < 0:
            print('Time out of bound')
            raise IndexError
        if data is not None:
            return np.asarray([self.allDset[data][i] for i in np.arange(index_begin, index_end)])
        else:
            return np.arange(index_begin, index_end)

if __name__ == "__main__":
    argv = 'bee1bis'
    video_path, h5_path = path_finder(argv)
    count, list_of_frame = video_to_frames(video_path)
    dset = Data_set(h5_path)

    plt.figure(1)
    plt.plot(dset.allDset['field_time_dataset'][:])
    plt.show()
    plt.figure(2)
    plt.plot(dset.allDset['image_time_dataset'][:])
    plt.show()






