
'''
Class FrameInfo includes Class BoxInfo
'''

class FrameInfo:
    def __init__(self, frame_id, boxes_info:list, path=''):
        self.frame_id = frame_id
        self.boxes_info:list[BoxInfo] = boxes_info
        self.ball_info:tuple = ()

    def add_box_info(self, box_info):
        self.boxes_info.append(box_info)

    def add_ball_position(self, ball_info:str):
        x, y = ball_info.split()
        self.ball_info = (int(x), int(y))


    def __len__(self):
        return len(self.boxes_info)

    def __iter__(self):
        return iter(self.boxes_info)

    def __repr__(self):
        return (f'Frame_ID: {self.frame_id}\n'
                f'No. Of Players: {len(self.boxes_info)}\n'
                f'Ball Position: {self.ball_info}\n\n')


class BoxInfo:
    def __init__(self, line):
        # split by space
        splits = line.split()
        (player_id, x_min,y_min,x_max,y_max, frame_id,
        lost, grouping, generated, annot) = splits

        self.player_id = int(player_id)
        self.bounding_box = int(x_min), int(y_min), int(x_max), int(y_max)
        self.frame_id = int(frame_id)
        self.lost = int(lost)
        self.grouping = int(grouping)
        self.generated = int(generated)
        self.category = annot


