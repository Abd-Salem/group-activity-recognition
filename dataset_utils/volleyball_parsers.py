import cv2
import os, pickle
from helper_utils.frame_box_info import BoxInfo, FrameInfo
from helper_utils.configs import CONFIG


def load_tracking_annotation(annot_path, ball_path=''):
    '''
    loading annotation file and parsing it line by line according to data annotation description
    :param annot_path: data annotation file path
    :param ball_path: ball center point file path
    :return frames_boxes:  dict contains frames that contain boxes
    '''
    with open(annot_path, 'r') as file:
        players_boxes = {i:[] for i in range(12)}
        frames_boxes = {}

        for line in file:
            box = BoxInfo(line)
            # ignore line if player id is false
            if box.player_id < 0 or box.player_id > 11:
                continue
            players_boxes[box.player_id].append(box)


        for player_id, boxes in players_boxes.items():
            # clipping frames to be only 9 frames
            boxes = boxes[6:-5]

            for box in boxes:
                if box.frame_id not in frames_boxes:
                    frames_boxes[box.frame_id] = []
                frames_boxes[box.frame_id].append(box)

        # dct of frames
        for frame_id, boxes in frames_boxes.items():
            frames_boxes[frame_id] = FrameInfo(frame_id, boxes)

    if ball_path:
        # get ball annotations and align frames with frames_boxes
        with open(ball_path, 'r') as file:
            frame_ball_position = file.readlines()
        frame_ball_position = frame_ball_position[10:30]
        frame_ball_position = frame_ball_position[6:-5]

        # add ball annotation for each frame
        for ball_pos, frame_id in zip(frame_ball_position, frames_boxes.keys()):
            frames_boxes[frame_id].add_ball_position(ball_pos)

    return frames_boxes


def load_clip_annotation(annot_path):
    '''
    reading annotation.txt file for each clip and get clip number and it's annotation
    :param annot_path: annotations.txt file path
    :return: clip_annot dct
    '''
    clip_annot = {}
    with open(annot_path, 'r') as file:
        for line in file:
            data = line.strip().split(' ')[0 : 2]
            target_frame = data[0].replace('.jpg', '')
            annot = data[1]
            clip_annot[target_frame] = annot

    return clip_annot



def load_volleyball_dataset(ball_info=False, config=None):
    '''
    loading players boxes info, players annotations, clips' labels and clips' paths in dict

    :param ball_info: load ball information in each frame
    :param config: configurations (default: CONFIG())
    :return: video annotation dct
    '''

    if config is None:
        config = CONFIG()


    # videos labels and frames information
    videos_annots = {}

    videos_ids = os.listdir(config.VIDEO_ROOT_DIR)
    videos_ids.sort()

    # loop over split
    for _, vid_id in enumerate(videos_ids):
        vid_dir_path = os.path.join(config.VIDEO_ROOT_DIR, vid_id)

        # Only dir. (skip files)
        if not os.path.isdir(vid_dir_path):
            continue

        # get clips annotations
        annot_path = os.path.join(vid_dir_path, 'annotations.txt')
        clip_annot_dct = load_clip_annotation(annot_path)

        clips_ids = os.listdir(vid_dir_path)
        clips_ids.sort(key=config.CUSTOM_KEY)

        clip_annot = {}

        # loop over clips of each video
        for _, clip_id in enumerate(clips_ids):
            clip_dir_path = os.path.join(vid_dir_path, clip_id)

            # Only dir. (skip files)
            if not os.path.isdir(clip_dir_path):
                continue

            # Aligning check
            assert clip_id in clip_annot_dct

            # get frame box info & ball info
            tracking_annot_path = os.path.join(config.TRACKING_ANNOTS_ROOT_DIR, vid_id, clip_id, f'{clip_id}.txt')
            ball_path = os.path.join(config.BALL_ROOT_DIR, vid_id, f'{clip_id}.txt')  if ball_info else ''
            frames_boxes = load_tracking_annotation(tracking_annot_path, ball_path)

            # group all annotation for each clip
            clip_annot[clip_id] = {
                'label' : clip_annot_dct[clip_id],
                'frames_boxes_dct' : frames_boxes,
                'clip_dir_path': clip_dir_path
            }
        # group clips for each video
        videos_annots[vid_id] = clip_annot

    return videos_annots


def handle_corrupted_dataset():
    '''
    check number of players for each frames in dataset and according to the percentage of corruption
    there'll be a suitable action
    :return: handled dataset
    '''

    # load videos for info validation
    videos_annots = load_volleyball_dataset()

    n_corrupted_clips, n_clips = 0, 0
    corrupted_clips_ids = set()
    for vid_id in videos_annots:
        # get clips num
        n_clips += len(videos_annots[vid_id])

        for clip_id in videos_annots[vid_id]:
            frames_boxes_dct = videos_annots[vid_id][clip_id]['frames_boxes_dct']
            for frame_info in frames_boxes_dct.values():
                if len(frame_info) != 12:
                    n_corrupted_clips += 1
                    corrupted_clips_ids.add(clip_id)
                    break

    if (n_corrupted_clips / n_clips) < 0.05:
        for vid_id in videos_annots:
            for clip_id in list(videos_annots[vid_id].keys()):
                if clip_id in corrupted_clips_ids:
                    del videos_annots[vid_id][clip_id]
    else:
        raise ValueError(f"Too many corrupted clips: {n_corrupted_clips}/{n_clips}")

    return videos_annots, corrupted_clips_ids


def load_clips_and_labels(split:list, image_level=True, config=None):
    '''
    align clips paths with their labels
    :param split: train, val, test
    :param image_level: full image or crops
    :param config: configurations
    :return: clips, labels, clips info (if crops)
    '''

    if config is None:
        config = CONFIG()

    videos_annots, corrupted_clips_ids = handle_corrupted_dataset()


    clips, labels, clips_info = [], [], []
    for vid_id in split:
        video_path = os.path.join(config.VIDEO_ROOT_DIR, vid_id)

        if not os.path.isdir(video_path):
            continue

        clips_ids = os.listdir(video_path)
        clips_ids.sort(key=config.CUSTOM_KEY)

        for clip_id in clips_ids:

            if clip_id in corrupted_clips_ids:
                continue

            if not os.path.isdir(os.path.join(video_path, clip_id)):
                continue

            label = videos_annots[vid_id][clip_id]['label']
            labels.append(label)

            frames_ids = list(videos_annots[vid_id][clip_id]['frames_boxes_dct'].keys())
            frames_ids.sort(key=config.CUSTOM_KEY)          # frames in each clip are in order

            if image_level:
                clip = [os.path.join(video_path, clip_id, f'{frame_id}.jpg') for frame_id in frames_ids]
                clips.append(clip)
            else:
                clip, clip_info = [], []
                for frame_id in frames_ids:
                    path = os.path.join(video_path, clip_id, f'{frame_id}.jpg')
                    clip.append(path)

                    # get frame info object and add it to clip info list
                    frame_info = videos_annots[vid_id][clip_id]['frames_boxes_dct'][frame_id]
                    frame_boxes_info = frame_info.boxes_info    # list of each frame's boxes info
                    clip_info.append(frame_boxes_info)

                clips.append(clip)
                clips_info.append(clip_info)

    if image_level:
        return clips, labels
    else:
        return clips, labels, clips_info

