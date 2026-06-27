import torch, cv2, os
from box_info import BoxInfo


dataset_root = '/Users/Abdallah Salem/Desktop/group-activity-recognition'

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


def visualize_clips(annot_path, video_path):
    '''
    visualize clipped frames with bounding boxes and annotations
    :param annot_path: annotation file path
    :param video_path: video frames path
    '''

    # get frames for each video
    frames_boxes = load_parse_annot_lines(annot_path)

    for frame_id, boxes in frames_boxes.items():
        img_path = os.path.join(video_path, f'{frame_id}.jpg')
        img = cv2.imread(img_path)

        # draw boxes and write annotations for each frame
        for box in boxes:
            x1,y1,x2,y2 = box.bounding_box
            cv2.rectangle(img, (x1, y1),(x2, y2), (0, 255, 0), 2)
            cv2.putText(img, box.annotation, (x1, y1-10),cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(300)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    annotation_path = f'{dataset_root}/volleyball_tracking_annotation/10/18360/18360.txt'
    video_path = f'{dataset_root}/videos_sample/10/18360/'
    visualize_clips(annotation_path, video_path)
