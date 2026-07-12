import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from volleyball_annot_loader_utils import load_tracking_annotation, dataset_root, custom_key, train_ids, val_ids

#   videos_annots['video_num']['clip_num']  -> frames_boxes dct contain each frame info  & annotations
#   frames_boxes_dct[frame_id]              -> Frame-info object contains: frame_id, list of boxes-info, ball info
#   Boxes_info[player_id]                   -> Box-Info object contains:player_id, frame_id, bounding-box, category



def check():
    '''
    make some checks: torch version ? , used device(cuda, cpu) ?, number of used devices ?
    '''
    print(f'Torch Version: {torch.__version__}')

    if torch.cuda.is_available():
        print('Cuda is available')

        device_num = torch.cuda.device_count()
        print(f'Device count: {device_num}')

        for i in range(device_num):
            print(f'Device {i}: {torch.get_device_name(i)}')

    else:
        print('Cuda is not available. Using CPU')

    current_device_name = torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'
    print(f'Current Device: {current_device_name}')



def load_extractor():
    '''
    Using resnet-50 architecture for feature extraction
    :return: extractorg
    '''
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    desired_arch = list(resnet.children())[:-1]           # drop fc layer
    extractor = torch.nn.Sequential(*desired_arch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor.to(device)        # load model to device
    extractor.eval()
    return extractor


def get_processor(full_image=False):
    '''
    get processor for preprocessing with respect to image level (full, crop)
    :param full_image: image level (full, crop)
    :return: processor
    '''
    if full_image:
        processor = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        processor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])
    return processor


def extract_features(videos_root, annot_root, output_root, model, full_image=False):
    '''
    extract representations for each frame or for each player of each frame and save them
    :param videos_root: videos root dir
    :param annot_root: annotations root dir
    :param output_root: output root dir
    :param model: pretrained network for feature extraction
    :param full_image: full image or crops
    '''


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    for _, vid_dir in enumerate(videos_dirs):
        vid_path = os.path.join(videos_root, vid_dir)

        if not os.path.isdir(vid_path):
            continue

        clips_dirs = os.listdir(vid_path)
        clips_dirs.sort(key=custom_key)

        for _, clip_dir in enumerate(clips_dirs):
            clip_path = os.path.join(vid_path, clip_dir)

            if not os.path.isdir(clip_path):
                continue

            annot_file = os.path.join(annot_root, vid_dir, clip_dir, f'{clip_dir}.txt')
            output_dir = os.path.join(output_root, vid_dir)

            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            frame_boxes = load_tracking_annotation(annot_file)
            with torch.no_grad():
                for frame_id, frame_info in frame_boxes.items():
                    # extract from target frames only
                    try:
                        frame_path = os.path.join(clip_path, f'{frame_id}.jpg')
                        img = Image.open(frame_path).convert('RGB')
                        processor = get_processor(full_image=full_image)

                        if full_image:
                            processed_img = processor(img).unsqueeze(0)
                            processed_img = processed_img.to(device)
                            repr = model(processed_img)
                            repr = repr.view(1, -1)

                        else:
                            processed_crops = []
                            for box_info in frame_info.boxes_info:
                                crop = img.crop(box_info.bounding_box)
                                processed_crop = processor(crop).unsqueeze(0)
                                processed_crops.append(processed_crop)
                            processed_img = torch.cat(processed_crops)
                            processed_img = processed_img.to(device)
                            repr = model(processed_img)
                            repr = repr.view(len(processed_img), -1)

                        # saving representations
                        output_path = os.path.join(output_dir, f'{frame_id}.npy')
                        np.save(output_path, repr.cpu().numpy())

                    except Exception as e:
                        print(f'Error: {e}')




if __name__ == '__main__':
    check()         # versions and machines

    # root paths
    videos_root = f'{dataset_root}/videos_sample'
    annot_root = f'{dataset_root}/volleyball_tracking_annotation'
    output_root = f'{dataset_root}/features/image-level/resnet'

    full_image = True      # full frame or crops

    model = load_extractor()        # extractor
    extract_features(videos_root, annot_root, output_root, model, full_image=full_image)   # extract features and save them
