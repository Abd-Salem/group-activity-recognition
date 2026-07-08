import os
import matplotlib.pyplot as plt
import torch, cv2
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from volleyball_annot_loader_utils import dataset_root, load_volleyball_dataset

#   videos_annots['video_num']['clip_num']  -> frames_boxes dct contain each frame info  & annotations
#   frames_boxes_dct[frame_id]              -> Frame-info object contains: frame_id, list of boxes-info, ball info
#   Boxes_info[player_id]                   -> Box-Info object contains:player_id, frame_id, bounding-box, category
#
videos_annots = load_volleyball_dataset()
videos_root = f'{dataset_root}/videos_sample'

processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

def prepare_data_extract_features(full_image=True, extractor=None):
    '''
    1- load videos following a custom schema (video_annotations dct structure)
    2- convert each image to RGB
    3- apply processor on dataset (resize, tensor, normalize)
    4- utilize data shape to be (B, CH, H, W)
    :return:
    '''

    prepared_dataset = {}
    extracted_features = {}

    # inference mode
    if extractor:
        extractor.eval()

    for vid_num in videos_annots:
        prepared_dataset[vid_num] = {}
        extracted_features[vid_num] = {}
        for clip_num in videos_annots[vid_num]:
            prepared_dataset[vid_num][clip_num] = {}
            extracted_features[vid_num][clip_num] = {}
            for frame_num in videos_annots[vid_num][clip_num]['frames_boxes_dct']:
                frame_path = os.path.join(videos_root, vid_num, clip_num, f'{frame_num}.jpg')
                frame = Image.open(frame_path).convert('RGB')

                if full_image:
                    processed_frame = processor(frame)
                    prepared_dataset[vid_num][clip_num][frame_num] = processed_frame
                    if extractor:
                        with extractor.no_grad():
                            extracted_features[vid_num][clip_num][frame_num] = extractor(processed_frame)
                else:
                    # crop -> transform -> stack
                    processed_cropped_boxes, extracted_features_boxes = [], []
                    for box in videos_annots[vid_num][clip_num]['frames_boxes_dct'][frame_num].boxes_info:
                        crop = frame.crop(box.bounding_box)
                        processed_crop = processor(crop)
                        processed_cropped_boxes.append(processed_crop)
                        if extractor:
                            with extractor.no_grad():
                                extracted_features[vid_num][clip_num][frame_num] = extractor(processed_crop)

                    prepared_dataset[vid_num][clip_num][frame_num] = processed_cropped_boxes
                    if extractor:
                        extracted_features[vid_num][clip_num][frame_num] = extracted_features_boxes
    if extractor:
        return prepared_dataset, extracted_features
    return prepared_dataset



def load_feature_extractor():
    '''
    load resnet and drop fc layers from architecture
    :return: extractor
    '''
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    new_arch = list(resnet.children())[:-1]          # drop last fc layer
    feature_extractor = torch.nn.Sequential(*new_arch)  # create feature extractor
    # freeze auto grad engine
    for param in feature_extractor.parameters():
        param.requires_grad = False

    return feature_extractor


if __name__ == '__main__':
    full_image = True       # extract features of full frame
    extractor = load_feature_extractor()
    prepared_dataset = prepare_data_extract_features(full_image=full_image, extractor=extractor)