import torch, cv2, os
from frame_box_info import BoxInfo, FrameInfo


dataset_root = '/Users/Abdallah Salem/Desktop/group-activity-recognition'

# Data relations:
#
#   - Each player presents in 20 frames for each video
#   - Each frame has annotations for 12 players
#

def load_parse_annot_lines(annot_path, ball_path):
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

        # dictionary of frames
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
    frames_boxes = load_parse_annot_lines(player_annot, ball_annot)

    for frame_id, frame in frames_boxes.items():
        img_path = os.path.join(video_frames, f'{frame_id}.jpg')
        img = cv2.imread(img_path)

        # draw boxes and ball and write annotations for each frame
        for box in frame:
            x1,y1,x2,y2 = box.bounding_box
            cv2.rectangle(img, (x1, y1),(x2, y2), (0, 255, 0), 2)
            cv2.putText(img, box.annotation, (x1, y1-10),cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        cv2.circle(img, (frame.ball_info[0], frame.ball_info[1]), 8, (0, 0, 255), -1)

        cv2.imshow('Image', img)
        cv2.waitKey(200)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    player_annot = f'{dataset_root}/volleyball_tracking_annotation/10/20500/20500.txt'
    ball_annot = f'{dataset_root}/volleyball_ball_annotation/10/20500.txt'
    video_frames = f'{dataset_root}/videos_sample/10/20500/'

    visualize_clips(player_annot, ball_annot, video_frames)