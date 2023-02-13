import moviepy.editor as mp
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import dlib
import glob
import cv2
import os
import imutils
from imutils import face_utils
import gdown
import numpy as np

def extract_audio(start_time, end_time, video_file_path, out_dir):
    make_dir(out_dir)
    my_clip = mp.VideoFileClip(video_file_path)
    my_clip = my_clip.subclip(start_time)
    video_file_path = video_file_path.split('/')[-1].replace('avi', 'wav')
    audio_file_path = os.path.join(out_dir, str(start_time)+video_file_path)
    my_clip.audio.write_audiofile(audio_file_path)
    return audio_file_path


def split_audio(audio_file_path, out_dir, min_silence_len=400, silence_thresh=-65):
    make_dir(out_dir)
    audio_name= audio_file_path.split('/')[-1].replace('.wav', '')
    sound = AudioSegment.from_mp3(audio_file_path)
    nonsilence_range = detect_nonsilent(sound, min_silence_len, silence_thresh)

    for i, chunk in enumerate(nonsilence_range):
        sound[chunk[0]:chunk[1]].export(os.path.join(out_dir, audio_name+'_'+str(i)+'.wav'), format="wav", bitrate="16k")
    return nonsilence_range


def split_video(video_file_path, out_dir, num, nonsilence_range):
    make_dir(out_dir)
    video_name = video_file_path.split('/')[-1].replace('.avi', '')
    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    expand_name = '.jpg'

    if not cap.isOpened():
        print("Please check the path.")

    count_frame = 0
    image_num = 0
    index = 0
    chunk = nonsilence_range[0]

    while 1:
        ret, frame = cap.read()
        count_frame += 1
        if (chunk[0] / 1000 * fps) <= count_frame <= (chunk[1] / 1000 * fps) and count_frame % num == 0:
            image_num += 1
            cv2.imwrite(os.path.join(out_dir, video_name + '_'+str(index) + '_'+str(image_num) + expand_name), frame)

        if count_frame > (chunk[1] / 1000 * fps) and index < len(nonsilence_range) - 1:
            index += 1
            chunk = nonsilence_range[index]

        if not ret:
            break


def lip_aspect_ratio(lip):
    # left top to left bottom
    A = np.linalg.norm(lip[2] - lip[9])  # 51, 59
    # right top to right bottom
    B = np.linalg.norm(lip[4] - lip[7])  # 53, 57
    # leftest to rightest
    C = np.linalg.norm(lip[0] - lip[6])  # 49, 55
    lar = (A + B) / (2.0 * C)

    return lar


def lip_motion_detection(video_path, DETECTOR):
    (LIPFROM, LIPTO) = (48, 68)
    HIGH_THRESHOLD = 0.49
    LOW_THRESHOLD = 0.4
    original_audio_dir = '../output/original_audio'
    segment_audio_dir = '../output/segment_audio'
    segment_image_dir = '../output/segment_image'

    # (_, tempfilename) = os.path.split(video_path)
    # (filename, _) = os.path.splitext(tempfilename)

    VC = cv2.VideoCapture(video_path)
    frame_num = 0
    start_frame = 0
    end_frame = 0

    fps = VC.get(cv2.CAP_PROP_FPS)

    while VC.isOpened():
        rval, frame = VC.read()
        if rval:
            frame_num += 1
            frame = imutils.resize(frame, width=640)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect face rect
            rects = DETECTOR(frame_gray, 0)

            if len(rects):
                rect = rects[0]
                # find key points inside the face rect
                shape = PREDICTOR(frame_gray, rect)
                shape = face_utils.shape_to_np(shape)

                # locate lip region
                lip = shape[LIPFROM:LIPTO]
                # get lip aspect ratio
                lar = lip_aspect_ratio(lip)

                if (lar > HIGH_THRESHOLD or lar < LOW_THRESHOLD):
                    if start_frame <= end_frame:
                        start_frame = frame_num
                elif start_frame > end_frame and frame_num - start_frame > fps:
                    end_frame = frame_num
                    print(start_frame, end_frame)
                    label = '/' + video_path.split('/')[-3]
                    audio_file_path = extract_audio(start_frame/fps, end_frame/fps, video, original_audio_dir + label)
                    audio_range = split_audio(audio_file_path, segment_audio_dir + label)
                    split_video(video, segment_image_dir + label, 5, audio_range)
                    start_frame = end_frame
            else:
                print('No face found!')
        else:
            break



def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == '__main__':
    video_list = glob.glob('../dataset/eNTERFACE/*/anger/*/*.avi')
    idx = 0

    model_path = '../checkpoints/shape_predictor_68_face_landmarks.dat'

    if not os.path.exists(model_path):
        # You can also click this link to download 'https://drive.google.com/file/d/1AwHKa2-QpcqkFgqTbOLoRoNBDVXfTZ05/view?usp=sharing'
        url = 'https://drive.google.com/uc?id=1AwHKa2-QpcqkFgqTbOLoRoNBDVXfTZ05'
        gdown.download(url, model_path, quiet=False)

    SHAPE_PREDICTOR = model_path
    DETECTOR = dlib.get_frontal_face_detector()
    PREDICTOR = dlib.shape_predictor(SHAPE_PREDICTOR)

    for video in video_list:
        idx+=1
        if idx%100==0:
            print(idx)
        lip_motion_detection(video, DETECTOR)
        # label = '/' + video.split('/')[-3]
        # audio_file_path = extract_audio(video, original_audio_dir + label)
        # audio_range = split_audio(audio_file_path, segment_audio_dir + label)
        # split_video(video, segment_image_dir + label, 5, audio_range)
