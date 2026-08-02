from helper_utils.frame_box_info import BoxInfo, FrameInfo


def load_tracking_annotation(annot_path, ball_path=None):
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