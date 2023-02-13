import numpy as np
from torch.utils.data import DataLoader
import pickle
import torch
import torch.utils.data as data
import glob
import cv2
import PIL.Image as Image


def combine_feature(dataset, data_dir):
    all_dataset = []
    emotion = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    for i in range(len(dataset["features"])):
        image_path = data_dir + '/segment_image/' + dataset["name"][i] + '*.jpg'
        image_list = glob.glob(image_path)

        image_array=[]
        for i in range(5):
            image = cv2.imread(image_list[i])
            image = cv2.resize(image, (256, 256))
            image = np.transpose(image, (2, 0, 1))
            image_array.append(image)

        all_dataset.append([dataset["features"][i], image_array, emotion.index(dataset["label"][i])])
    return all_dataset


def combine_feature_with_four_images(dataset, data_dir):
    all_dataset = []
    emotion = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    for i in range(len(dataset["features"])):
        image_path = data_dir + '/segment_image/' + dataset["name"][i] + '*.jpg'

        image_list = glob.glob(image_path)
        to_image = Image.new('RGB', (256, 256))

        for y in range(1, 3):
            for x in range(1, 3):
                from_image = Image.open(image_list[(y - 1) * 2 + x])
                from_image = from_image.resize((128, 128), Image.ANTIALIAS)
                to_image.paste(from_image, ((x - 1) * 128, (y - 1) * 128))

        to_image.save(dataset["name"][i] + '.jpg')
        image_array = np.transpose(to_image, (2, 0, 1))

        all_dataset.append([dataset["features"][i], image_array, emotion.index(dataset["label"][i])])
    return all_dataset


def sort_data(data_set):
    indices = sorted(range(len(data_set)),
                     key=lambda k: len(data_set[k][0][0]),
                     reverse=True)
    data_set = [data_set[i] for i in indices]
    return data_set, indices


def pad_tensor(vec, pad, dim):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size).cuda()], dim=dim)


def collate_fn(instances):
    max_len = 512
    batch = []
    for (feature, image, label) in instances:
        batch.append((pad_tensor(feature, pad=max_len, dim=1), image, label))

    f = list(map(lambda x: x[0], batch))
    i = list(map(lambda x: x[1], batch))
    l = list(map(lambda x: x[2], batch))
    features = torch.stack(f, dim=0)
    images = torch.Tensor(i)
    labels = torch.Tensor(l)
    return (features, images, labels)


def get_dataloader(data_dir, batch_size=16, four_images=False):
    audio_features = pickle.load(open(data_dir + '/audio_feature.pkl', 'rb'))
    if four_images:
        dataset = combine_feature(audio_features, data_dir)
    else:
        dataset = combine_feature_with_four_images(audio_features, data_dir)

    data_size = len(dataset)

    dataset_sizes = {"train": int(data_size * 0.8), "test": data_size - int(data_size * 0.8)}

    train_set, test_set = data.random_split(dataset, [dataset_sizes["train"], dataset_sizes["test"]])

    sorted_train, train_indices = sort_data(train_set)
    sorted_test, test_indices = sort_data(test_set)

    trains = torch.utils.data.DataLoader(sorted_train, batch_size=batch_size, num_workers=0, drop_last=True,
                                         collate_fn=collate_fn)
    tests = torch.utils.data.DataLoader(sorted_test, batch_size=batch_size, num_workers=0, drop_last=True,
                                        collate_fn=collate_fn)

    return trains, tests
