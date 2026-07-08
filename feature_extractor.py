import os
import matplotlib.pyplot as plt
import torch, cv2
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from volleyball_annot_loader_utils import load_tracking_annotation, dataset_root

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


def extract_features(clip_path, annot_path, model, output_file, full_image=False):
    frame_boxes = load_tracking_annotation(annot_path)

    with model.no_grad():
        for frame_id, frame_info in frame_boxes.items():
            try:
                frame_path = os.path.join(clip_path, f'{frame_id}.jpg')
                img = Image.open(frame_path).convert('RGB')
                processor = get_processor(full_image=full_image)

                if full_image:
                    processed_img = processor(img).unsqueeze(0)
                    repr = model(processed_img)
                    repr = repr.view(1, -1)

                else:
                    processed_crops = []
                    for box_info in frame_info.boxes_info:
                        crop = img.crop(box_info.bounding_box)
                        processed_crop = processor(crop).unsqueeze(0)
                        processed_crops.append(processed_crop)
                    processed_img = torch.cat(processed_crops)
                    repr = model(processed_img)
                    repr = repr.view(len(processed_img), -1)

            except Exception as e:
                print(f'Error: {e}')




if __name__ == '__main__':
    check()

    videos_root = f'{dataset_root}/videos'
    annot_root = f'{dataset_root}/volleyball_tracking_annotation'
    output_root = f'{dataset_root}/features/image-level/resnet'
