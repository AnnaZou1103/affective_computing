import torch
import fairseq
import librosa
import pickle
import glob
from typing import Any, Dict
import re

if __name__ == '__main__':
    input_path = '../output/segment_audio/*/*.wav'
    output_path = '../output/audio_feature.pkl'
    checkpoint = '../checkpoints/wav2vec_large.pt'  # path to wav2vec_large model
    aggregate = 'none'
    context = True

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint])
    model = model[0]
    model.cuda()
    model.eval()
    torch.set_grad_enabled(False)

    new_features = []
    names = []
    labels = []
    storewav = []
    storez = []

    for file in glob.glob(input_path):
        label = file.split('/')[-2]
        name = file.split('/')[-1]
        name = re.findall(r's\d*_.*_', name)[0]

        wav, rate = librosa.load(file, sr=16000, mono=True, res_type="kaiser_fast")
        tensor = torch.tensor(wav, device="cuda").unsqueeze(0)
        z = model.feature_extractor(tensor)
        if model.vector_quantizer is not None:
            z, _ = model.vector_quantizer.forward_idx(z)
        feats = model.feature_aggregator(z).squeeze(0) if context else z

        new_features.append(feats)
        names.append(name)
        labels.append(label)

    data: Dict[str, Any] = {}
    data["name"] = names
    data["features"] = new_features
    data["label"] = labels

    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=4)