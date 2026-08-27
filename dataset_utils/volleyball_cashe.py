from volleyball_builders import load_volleyball_dataset
from configs import CONFIG
import os, pickle

def save_annotations(ball_info=False, config=None):
    '''
    save all annotations in pickle file version
    :param ball_info: get or ignore ball information
    '''
    if config is None:
        config = CONFIG()

    videos_annots = load_volleyball_dataset(ball_info=ball_info)

    if not os.path.isdir(config.ANNOT_SAVE_DIR):
        os.makedirs(config.ANNOT_SAVE_DIR)


    with open(f'{config.ANNOT_SAVE_DIR}/annots.pickle', 'wb') as file:
        pickle.dump(videos_annots, file)

def load_annotations(config=None):
    '''
    load saved annotations
    '''

    if config is None:
        config = CONFIG()

    save_file = f'{config.ANNOT_SAVE_DIR}/annots.pickle'
    with open(save_file, 'rb') as file:
        videos_annots = pickle.load(file)

    return videos_annots