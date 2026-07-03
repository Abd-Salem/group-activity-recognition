import torch, cv2
import os, pickle
from frame_box_info import BoxInfo, FrameInfo


dataset_root = '/Users/Abdallah Salem/Desktop/group-activity-recognition'
custom_key = lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)

# Data relations:
#
#   - Each player presents in 20 frames for each video
#   - Each frame has annotations for 12 players
#

def load_tracking_annotation(annot_path, ball_path):
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
            boxes = boxes[5:-6]

            for box in boxes:
                if box.frame_id not in frames_boxes:
                    frames_boxes[box.frame_id] = []
                frames_boxes[box.frame_id].append(box)

        # dct of frames
        for frame_id, boxes in frames_boxes.items():
            frames_boxes[frame_id] = FrameInfo(frame_id, boxes)

    # get ball annotations and align frames with frames_boxes
    with open(ball_path, 'r') as file:
        frame_ball_position = file.readlines()
    frame_ball_position = frame_ball_position[10:30]
    frame_ball_position = frame_ball_position[5:-6]

    # add ball annotation for each frame
    for ball_pos, frame_id in zip(frame_ball_position, frames_boxes.keys()):
        frames_boxes[frame_id].add_ball_position(ball_pos)

    return frames_boxes


def visualize_clips(player_annot, ball_annot, video_frames):
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


def load_volleyball_dataset():
    '''
    Dataset will be loaded from two different paths then it'll be gathered in dct obj
        - clip annotation from annotation.txt file in video dir
        - Frame Boxes Info & Ball Info from tracking annotations dir and ball annotations dir
    :return: video annotation dct
    '''

    # roots
    video_root = os.path.join(dataset_root, 'videos')
    annot_root = os.path.join(dataset_root, 'volleyball_tracking_annotation')
    ball_root = os.path.join(dataset_root, 'volleyball_ball_annotation')

    # All Annotations
    videos_annots = {}


    vid_dirs_names = os.listdir(video_root)
    vid_dirs_names.sort(key=custom_key)

    # loop over videos
    for _, vid_dir_name in enumerate(vid_dirs_names):
        vid_dir_path = os.path.join(video_root, vid_dir_name)

        # Only dir. (skip files)
        if not os.path.isdir(vid_dir_path):
            continue

        # get clips annotations
        annot_path = os.path.join(vid_dir_path, 'annotations.txt')
        clip_annot_dct = load_clip_annotation(annot_path)

        clips_dirs_names = os.listdir(vid_dir_path)
        clips_dirs_names.sort(key=custom_key)

        clip_annot = {}

        # loop over clips of each video
        for _, clip_dir_name in enumerate(clips_dirs_names):
            clip_dir_path = os.path.join(vid_dir_path, clip_dir_name)

            # Only dir. (skip files)
            if not os.path.isdir(clip_dir_path):
                continue

            # Aligning check
            assert clip_dir_name in clip_annot_dct

            # get frame box info & ball info
            tracking_annot_path = os.path.join(annot_root, vid_dir_name, clip_dir_name, f'{clip_dir_name}.txt')
            ball_path = os.path.join(ball_root, vid_dir_name, f'{clip_dir_name}.txt')
            frames_boxes = load_tracking_annotation(tracking_annot_path, ball_path)

            # group all annotation for each clip
            clip_annot[clip_dir_name] = {
                'annotations' : clip_annot_dct[clip_dir_name],
                'frames_boxes_dct' : frames_boxes
            }
        # group clips for each video
        videos_annots[vid_dir_name] = clip_annot

    return videos_annots


def create_pickle_version():
    '''
    save all annotations in pickle file version
    '''
    videos_annots = load_volleyball_dataset()
    with open(f'{dataset_root}/all_annotations.pickle', 'wb') as file:
        pickle.dump(videos_annots, file)


def test_pickle_version():
    '''
    Just testing pickle file
    '''

    with open(f'{dataset_root}/all_annotations.pickle', 'rb') as file:
        videos_annots = pickle.load(file)

    frame : FrameInfo = videos_annots['0']['13456']['frames_boxes_dct']
    print(f'Frame ID: {frame[13456].frame_id}')
    print(f'Ball Position: {frame[13456].ball_info}')
    print(f'Player ID: {frame[13456].boxes_info[0].player_id}')
    print(f'Player position: {frame[13456].boxes_info[0].bounding_box}')
    print(f'Player category: {frame[13456].boxes_info[0].category}')


if __name__ == '__main__':
    player_annot = f'{dataset_root}/volleyball_tracking_annotation/7/51725/51725.txt'
    ball_annot = f'{dataset_root}/volleyball_ball_annotation/7/51725.txt'
    video_frames = f'{dataset_root}/videos_sample/7/51725/'

    visualize_clips(player_annot, ball_annot, video_frames)