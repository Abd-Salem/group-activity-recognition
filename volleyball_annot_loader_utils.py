import cv2
import os, pickle
from frame_box_info import BoxInfo, FrameInfo
from configs import CONFIG


config = CONFIG()
custom_key = lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)

# Data relations:
#
#   - Each player presents in 20 frames for each video
#   - Each frame has annotations for 12 players
#

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


def visualize_clips(player_annot, video_frames, ball_annot=''):
    '''
    visualize clipped frames with bounding boxes and annotations
    :param player_annot: path of annotation for each player
    :param ball_annot: path of annotation of the ball inside each frame
    :param video_frames: video frames path
    '''

    # get frames for each video
    frames_boxes = load_tracking_annotation(player_annot, ball_annot)

    for frame_id, frame in frames_boxes.items():
        img_path = os.path.join(video_frames, f'{frame_id}.jpg')
        img = cv2.imread(img_path)

        # draw boxes and ball and write annotations for each frame
        for box in frame:
            x1,y1,x2,y2 = box.bounding_box
            cv2.rectangle(img, (x1, y1),(x2, y2), (0, 255, 0), 2)
            cv2.putText(img, box.category, (x1, y1-10),cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        if ball_annot:
            cv2.circle(img, (frame.ball_info[0], frame.ball_info[1]), 8, (0, 0, 255), -1)

        cv2.imshow('Image', img)
        cv2.waitKey(200)
    cv2.destroyAllWindows()


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


def load_volleyball_dataset(ball_info=False):
    '''
    Dataset will be loaded from two different paths then it'll be gathered in dct obj
        - clip annotation from annotation.txt file in video dir
        - Frame Boxes Info & Ball Info from tracking annotations dir and ball annotations dir
    :return: video annotation dct
    '''


    # videos labels and frames information
    videos_annots = {}

    vid_dirs = os.listdir(config.VIDEO_ROOT_DIR)
    vid_dirs.sort()

    # loop over split
    for _, vid_dir in enumerate(vid_dirs):
        vid_dir_path = os.path.join(config.VIDEO_ROOT_DIR, vid_dir)

        # Only dir. (skip files)
        if not os.path.isdir(vid_dir_path):
            continue

        # get clips annotations
        annot_path = os.path.join(vid_dir_path, 'annotations.txt')
        clip_annot_dct = load_clip_annotation(annot_path)

        clip_dirs = os.listdir(vid_dir_path)
        clip_dirs.sort(key=custom_key)

        clip_annot = {}

        # loop over clips of each video
        for _, clip_dir in enumerate(clip_dirs):
            clip_dir_path = os.path.join(vid_dir_path, clip_dir)

            # Only dir. (skip files)
            if not os.path.isdir(clip_dir_path):
                continue

            # Aligning check
            assert clip_dir in clip_annot_dct

            # get frame box info & ball info
            tracking_annot_path = os.path.join(config.TRACKING_ANNOTS_DIR, vid_dir, clip_dir, f'{clip_dir}.txt')
            ball_path = os.path.join(config.BALL_ROOT_DIR, vid_dir, f'{clip_dir}.txt')  if ball_info else ''
            frames_boxes = load_tracking_annotation(tracking_annot_path, ball_path)

            # group all annotation for each clip
            clip_annot[clip_dir] = {
                'label' : clip_annot_dct[clip_dir],
                'frames_boxes_dct' : frames_boxes,
                'clip_dir_path': clip_dir_path,
            }
        # group clips for each video
        videos_annots[vid_dir] = clip_annot

    return videos_annots


def save_annotations(ball_info=False):
    '''
    save all annotations in pickle file version
    :param ball_info: get or ignore ball information
    '''
    videos_annots = load_volleyball_dataset(ball_info=ball_info)

    if not os.path.isdir(config.ANNOT_ROOT_DIR):
        os.makedirs(config.ANNOT_ROOT_DIR)


    with open(f'{config.ANNOT_ROOT_DIR}/annots.pickle', 'wb') as file:
        pickle.dump(videos_annots, file)


def load_annotations():
    '''
    load saved annotations
    '''

    save_file = f'{config.ANNOT_ROOT_DIR}/annots.pickle'
    with open(save_file, 'rb') as file:
        videos_annots = pickle.load(file)

    return videos_annots


if __name__ == '__main__':
    # testing case
    player_annot = f'{config.DATASET_ROOT_DIR}/volleyball_tracking_annotation/7/51725/51725.txt'
    ball_annot = f'{config.BALL_ROOT_DIR}/7/51725.txt'
    video_frames = f'{config.DATASET_ROOT_DIR}/videos_sample/7/51725/'

    visualize_clips(player_annot, video_frames, ball_annot)