from volleyball_loaders import load_tracking_annotation
import cv2, os
from helper_utils.configs import CONFIG



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


if __name__ == '__main__':
    config = CONFIG()
    # testing case
    player_annot = f'{config.TRACKING_ANNOTS_ROOT_DIR}/7/51725/51725.txt'
    ball_annot = f'{config.BALL_ROOT_DIR}/7/51725.txt'
    video_frames = f'{config.DATASET_ROOT_DIR}/samples/videos/7/51725/'

    visualize_clips(player_annot, video_frames, ball_annot)