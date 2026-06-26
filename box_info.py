
'''
Class of Box's Information that's gotten from each line
'''
class BoxInfo:
    def __init__(self, line):
        # split by space
        splits = line.split()
        (player_id, x_min,y_min,x_max,y_max, frame_id,
        lost, grouping, generated, annot) = splits

        self.player_id = player_id
        self.bounding_box = x_min, y_min, x_max, y_max
        self.frame_id = frame_id
        self.lost = lost
        self.grouping = grouping
        self.generated = generated
        self.annotation = annot


