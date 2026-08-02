import os
from helper_utils.configs import  CONFIG
from volleyball_parsers import load_clip_annotation, load_tracking_annotation


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
            ball_path = os.path.join(config.BALL_ROOT_DIR, vid_id, f'{clip_id}.txt')  if ball_info else None
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