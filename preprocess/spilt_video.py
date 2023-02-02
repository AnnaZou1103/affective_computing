import moviepy.editor as mp
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import glob
import cv2
import os

def extract_audio(video_file_path, out_dir):
    make_dir(out_dir)
    my_clip = mp.VideoFileClip(video_file_path)
    video_file_path = video_file_path.split('/')[-1].replace('avi', 'wav')
    audio_file_path = os.path.join(out_dir, video_file_path)
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


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == '__main__':
    video_list = glob.glob('../dataset/eNTERFACE/*/anger/*/*.avi')
    original_audio_dir = '../output/original_audio'
    segment_audio_dir = '../output/segment_audio'
    segment_image_dir = '../output/segment_image'
    idx=0

    for video in video_list:
        idx+=1
        if idx%100==0:
            print (idx)

        label = '/'+video.split('/')[-3]
        audio_file_path = extract_audio(video, original_audio_dir + label)
        audio_range = split_audio(audio_file_path, segment_audio_dir + label)
        split_video(video, segment_image_dir + label, 5, audio_range)
