
'''
Class of Box's Information that's gotten from each line
'''
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
        self.annotation = annot


