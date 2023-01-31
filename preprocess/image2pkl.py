import cv2
import glob
from typing import Any, Dict
import pickle

if __name__ == '__main__':
    image_list = glob.glob('../output/segment_image/*/*.jpg')
    output_path = '../output/image_feature.pkl'

    arrays = []
    names = []
    labels = []

    for image in image_list:
        name = image.split('/')[-1].replace('.jpg','')
        label = image.split('/')[-2]
        image_array = cv2.imread(image)

        arrays.append(image_array)
        names.append(name)
        labels.append(label)

    data: Dict[str, Any] = {}
    data["name"] = names
    data["features"] = arrays
    data["label"] = labels

    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=4)