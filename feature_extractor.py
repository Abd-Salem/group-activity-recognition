import os.path
import matplotlib.pyplot as plt
import torch, cv2
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from torchvision.models import ResNet50_Weights
from torchvision.models.quantization import resnet50
from volleyball_annot_loader_utils import dataset_root


#   videos_annots['video_num']['clip_num']  -> frames_boxes dct contain each frame info  & annotations
#   frames_boxes_dct[frame_id]              -> Frame-info object contains: frame_id, list of boxes-info, ball info
#   Boxes_info[player_id]                   -> Box-Info object contains:player_id, frame_id, bounding-box, category
#
videos_annots = {}
videos_root = f'{dataset_root}/videos'

processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),   # <-- this line already does the /255 scaling
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

def prepare_data(full_image=True):
    '''
    1- load videos following a custom schema (video_annotations dct structure)
    2- convert each image to RGB
    3- apply processor on dataset (resize, tensor, normalize)
    4- utilize data shape to be (B, CH, H, W)
    :return:
    '''

    prepared_dataset = {}
    for vid_num in videos_annots:
        prepared_dataset[vid_num] = {}
        for clip_num in videos_annots[vid_num]:
            prepared_dataset[vid_num][clip_num] = {}
            for frame_num in videos_annots[vid_num][clip_num]['frames_boxes_dct']:
                frame_path = os.path.join(videos_root, vid_num, clip_num, f'{frame_num}.jpg')
                print(frame_path)
                frame = Image.open(frame_path).convert('RGB')

                if full_image:
                    processed_frame = processor(frame)
                    prepared_dataset[vid_num][clip_num][frame_num] = processed_frame
                else:
                    # crop -> transform -> stack
                    processed_cropped_boxes = []
                    for box in videos_annots[vid_num][clip_num]['frames_boxes_dct'][frame_num].boxes_info:
                        crop = frame.crop(box.bounding_box)
                        processed_crop = processor(crop)
                        processed_cropped_boxes.append(processed_crop)
                    prepared_dataset[vid_num][clip_num][frame_num] = processed_cropped_boxes
    return prepared_dataset.values()
