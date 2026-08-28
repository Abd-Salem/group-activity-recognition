from configs import CONFIG
from PIL import Image
import os, torch
from dataset_utils.volleyball_datasets import PersonLevelDataset, ImageLevelDataset
from dataset_utils.volleyball_builders import load_clips_and_labels
from helper_utils.feature_extraction import get_processor
from torch.utils.data import DataLoader
import pytest


def validate_volleyball_dataset(clips, labels, image_level=True, clips_info=None, config=None):
    '''
    check loaded data validity before training or evaluating
    :param clips: clips
    :param labels: labels
    :param clips_info: clips information if image-level is False
    :param config: configurations
    :return: True, empty list if there is no any issue
    :return: False, list of issues if there are some issues found
    '''
    if config is None:
        config = CONFIG()

    issues = []

    for clip_idx, (clip, label) in enumerate(zip(clips, labels)):

        # check labels
        if label not in config.LABELS:
            issues.append(f'Clip {clip_idx}: label {label} is not in config.LABELS')
            continue

        for frame_idx, path in enumerate(clip):
            # check paths
            if not os.path.exists(path):
                issues.append(f'Clip {clip_idx}|frame{frame_idx}: {path} does not exist !!')
                continue

            # check file readability
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    img.verify()
            except Exception as e:
                issues.append(f'Clip({clip_idx}) | frame({frame_idx}) | Path( {path} ) | Exception( {e} )')
                continue

            # person level checking
            if not image_level and clips_info is not None:
                frame_boxes = clips_info[clip_idx][frame_idx]

                if len(frame_boxes) != 12:
                    issues.append(f'Clip({clip_idx}) | Frame({frame_idx}) | Path( {path} ) | Error( Number of player boxes are not 12 )')
                    continue

                # check bounding boxes
                for box_info in frame_boxes:
                    x1, y1, x2, y2 = box_info.bounding_box

                    if x1 >= x2 or y1 >= y2:
                        issues.append(f'Clip({clip_idx}) | Frame({frame_idx}) | Path( {path} ) | Error( Invalid bounding box )')
                    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                        issues.append(f'Clip({clip_idx}) | Frame({frame_idx}) | Path( {path} )  | Error (Bounding box out of frame bounds )')

    # print issues if there are some
    if issues:
        if len(issues) > 20:
            for issue in issues[:20]:
                print(f'Validation {issue}')
            print(f'... and {len(issues) - 20} more')
        else:
            for issue in issues:
                print(f'Validation {issue}')

        return False, issues
    else:
        print(f'No Validation Errors')
        return True, []



@pytest.fixture
def dataset_loader():
    '''returns a function to build a person level dataset loader, temporal or not'''
    def _make_loader(temporal=True, full_image=True):
        config = CONFIG()
        processor = get_processor(full_image=full_image)
        clips, labels, clips_info = load_clips_and_labels(
            split=config.TRAIN_IDS, image_level=full_image, config=config
        )

        if full_image:
            dataset = ImageLevelDataset(
                paths=clips, labels=labels, processor=processor,
                temporal=temporal
            )
        else:
            dataset = PersonLevelDataset(
                paths=clips, labels=labels, clips_info=clips_info,
                processor=processor, temporal=temporal
            )

        return DataLoader(dataset=dataset, batch_size=2, shuffle=True)
    return _make_loader


def test_person_level_dataset_temporal(dataset_loader):
    '''testing shape and datatype of clips and labels loaded using PersonalLevelDataset class (temporal)'''
    loader = dataset_loader(temporal=True, full_image=False)
    clip, label = next(iter(loader))

    assert isinstance(clip, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    assert clip.shape == (2, 9, 12, 3, 255, 255)
    assert label.shape == (2, )


def test_person_level_dataset_not_temporal(dataset_loader):
    '''testing shape and datatype of clips and labels loaded from PersonLevelDataset class (not temporal) '''
    loader = dataset_loader(temporal=False, full_image=False)
    clip, label = next(iter(loader))

    assert isinstance(clip, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    assert clip.shape == (2, 12, 3, 255, 255)
    assert label.shape == (2, )


def test_image_level_dataset_temporal(dataset_loader):
    '''testing shape and datatype of clips and labels loaded from ImageLevelDataset class (temporal) '''
    loader = dataset_loader(temporal=True, full_image=True)
    clip, label = next(iter(loader))

    assert isinstance(clip, torch.Tensor)
    assert isinstance(label, torch.Tensor)    

    assert clip.shape == (2, 9, 3, 224, 224)
    assert label.shape == (2, )


def test_image_level_dataset_not_temporal(dataset_loader):
    '''testing shape and datatype of clips and labels loaded from ImageLevelDataset class (not temporal) '''    
    loader = dataset_loader(temporal=False, full_image=True)
    clip, label = next(iter(loader))

    assert isinstance(clip, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    assert clip.shape == (2, 3, 224, 224)
    assert label.shape == (2, )



