import os
import cv2
import glob
import numpy as np
import argparse
import pdb


def extract_opencv(filename):
    video = []
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read() # BGR
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    video = np.array(video)
    return video[...,::-1]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    
    args = get_args()
    data_folder = args.data_folder

    filenames = glob.glob(data_folder+'/*.mp4')
    filenames = sorted(filenames)
    i=1
    for filename in filenames:
        print(i, len(filenames))
        data = extract_opencv(filename) 
        path_to_save = os.path.join(data_folder.replace('Video','Video_npy'),
                                    filename.split('/')[-1][:-4]+'.npy')
        if not os.path.exists(os.path.dirname(path_to_save)):
            try:
                os.makedirs(os.path.dirname(path_to_save))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        np.save(path_to_save, data)
        i+=1