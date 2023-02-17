import moviepy.editor as mp
import dlib
import glob
import cv2
import os
import imutils
from imutils import face_utils
import gdown
import numpy as np


def extract_audio(lip_motion_range, video_file_path, out_dir):
    make_dir(out_dir)
    my_clip = mp.VideoFileClip(video_file_path)
    for audio_range in lip_motion_range:
        start_time = audio_range[0]
        end_time = audio_range[1]
        sub_clip = my_clip.subclip(start_time, end_time)
        audio_file_path = video_file_path.split('/')[-1].replace('.avi', '') + '_'+str(lip_motion_range.index(audio_range))+ '_' + str(start_time) + '.wav'
        audio_file_path = os.path.join(out_dir, audio_file_path)
        sub_clip.audio.write_audiofile(audio_file_path)


def lip_aspect_ratio(lip):
    # left top to left bottom
    A = np.linalg.norm(lip[2] - lip[9])
    # right top to right bottom
    B = np.linalg.norm(lip[4] - lip[7])
    # leftest to rightest
    C = np.linalg.norm(lip[0] - lip[6])
    lar = (A + B) / (2.0 * C)

    return lar


def lip_motion_detection(video_path, out_dir, detector, predictor, expand_name='.jpg', frame_frequency=5):
    make_dir(out_dir)
    (LIPFROM, LIPTO) = (48, 68)
    HIGH_THRESHOLD = 0.49
    LOW_THRESHOLD = 0.4

    VC = cv2.VideoCapture(video_path)
    frame_num = 0
    start_frame = 0
    end_frame = 0
    index = 0
    lip_motion_range = []

    fps = VC.get(cv2.CAP_PROP_FPS)

    while VC.isOpened():
        rval, frame = VC.read()
        if rval:
            frame_num += 1
            frame = imutils.resize(frame, width=640)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(frame_gray, 0)

            if len(rects):
                rect = rects[0]
                # find key points inside the face rect
                shape = predictor(frame_gray, rect)
                shape = face_utils.shape_to_np(shape)

                # locate lip region
                lip = shape[LIPFROM:LIPTO]
                lar = lip_aspect_ratio(lip)

                if lar > HIGH_THRESHOLD or lar < LOW_THRESHOLD:
                    if start_frame <= end_frame:
                        start_frame = frame_num

                    if (frame_num - start_frame) % frame_frequency == 0:
                        cv2.imwrite(
                            os.path.join(out_dir,
                                         video_path.split('/')[-1].replace('.avi', '') + '_' + str(index) + '_' + str(
                                             frame_num) + expand_name), frame)

                elif start_frame > end_frame and frame_num - start_frame > fps:
                    index += 1
                    end_frame = frame_num
                    lip_motion_range.append([start_frame / fps, end_frame / fps])
                    start_frame = end_frame
            else:
                print('No face found!')
        else:
            break

    return lip_motion_range


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == '__main__':
    video_list = glob.glob('../dataset/eNTERFACE/subject 1/*/*/*.avi')
    idx = 0

    model_path = '../checkpoints/shape_predictor_68_face_landmarks.dat'
    segment_audio_dir = '../output/segment_audio'
    segment_image_dir = '../output/segment_image'

    if not os.path.exists(model_path):
        # You can also click this link to download 'https://drive.google.com/file/d/1AwHKa2-QpcqkFgqTbOLoRoNBDVXfTZ05/view?usp=sharing'
        url = 'https://drive.google.com/uc?id=1AwHKa2-QpcqkFgqTbOLoRoNBDVXfTZ05'
        gdown.download(url, model_path, quiet=False)

    SHAPE_PREDICTOR = model_path
    DETECTOR = dlib.get_frontal_face_detector()
    PREDICTOR = dlib.shape_predictor(SHAPE_PREDICTOR)

    for video in video_list:
        idx += 1
        if idx % 100 == 0:
            print(idx)

        label = '/' + video.split('/')[-3]
        lip_motion_range = lip_motion_detection(video, segment_image_dir + label, DETECTOR, PREDICTOR)
        extract_audio(lip_motion_range, video, segment_audio_dir + label)
