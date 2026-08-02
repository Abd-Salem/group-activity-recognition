from helper_utils.configs import CONFIG
from PIL import Image
import os


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
                issues.append(f'Clip {clip_idx}/frame{frame_idx}: {path} does not exist !!')
                continue

            # check file readability
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    img.verify()
            except Exception as e:
                issues.append(f'Clip({clip_idx}) / frame({frame_idx}) / Path( {path} ) / Exception( {e} )')
                continue

            # person level checking
            if not image_level and clips_info is not None:
                frame_boxes = clips_info[clip_idx][frame_idx]

                if len(frame_boxes) != 12:
                    issues.append(f'Clip({clip_idx}) / Frame({frame_idx}) / Path( {path} ) / Error( Number of player boxes are not 12 )')
                    continue

                # check bounding boxes
                for box_info in frame_boxes:
                    x1, y1, x2, y2 = box_info.bounding_box

                    if x1 >= x2 or y1 >= y2:
                        issues.append(f'Clip({clip_idx})/Frame({frame_idx}) / Path( {path} ) / Error( Invalid bounding box )')
                    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                        issues.append(f'Clip({clip_idx})/Frame({frame_idx}) / Path( {path} )  / Error (Bounding box out of frame bounds )')

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