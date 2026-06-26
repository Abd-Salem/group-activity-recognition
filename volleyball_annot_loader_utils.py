import torch, cv2
from box_info import BoxInfo


# Data relations:
#
#   - Each player presents in 20 frames for each video
#   - Each frame has annotations for 12 players
#

def load_parse_annot_lines(path):
    '''
    loading annotation file and parsing it line by line according to data annotation description
    :param path: data annotation file path
    :return frames_boxes:  dict contains 9 frames with annotations and bounding boxes' coordinates for each player
    '''
    with open(path, 'r') as file:
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

    return frames_boxes



