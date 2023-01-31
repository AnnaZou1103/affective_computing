## Usage
### Preprocess
To preprocess the eNTERFACE dataset:
1. Go to /dataset.
2. Download the eNTERFACE dataset and add it to a new folder called eNTERFACE.
3. Run preprocess/split_video.py file to split the audio and segment the video into a series of images. The outputs are stored in /output.

To extract the audio feature:
1. Go to /checkpoints.
2. Download [wav2vec_large.pt](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt). 
3. Run preprocess/audio_feature_extraction.py. The extracted audio features are stored in /output.

### Train
Run train.py.
