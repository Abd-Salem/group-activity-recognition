import yaml, os

class CONFIG:
    '''
    preparing configurations with respect to yaml file
    '''
    def __init__(self, path='../helper_utils/configs.yml'):
        with open(path, 'r') as file:
            data = yaml.safe_load(file)

        self.DATASET_ROOT_DIR = data['dataset_root']
        self.VIDEO_ROOT_DIR = f'{self.DATASET_ROOT_DIR}/samples/videos'
        self.ANNOT_ROOT_DIR = f'{self.DATASET_ROOT_DIR}/annotations'
        self.BALL_ROOT_DIR = f'{self.ANNOT_ROOT_DIR}/volleyball_ball_annotation'
        self.TRACKING_ANNOTS_ROOT_DIR = f'{self.ANNOT_ROOT_DIR}/volleyball_tracking_annotation'
        self.FEATURES_ROOT_DIR = f'{self.DATASET_ROOT_DIR}/samples/features'

        self.ANNOT_SAVE_DIR = f'{self.ANNOT_ROOT_DIR}/all-annotations'
        self.IMAGE_LEVEL_DIR = f'{self.FEATURES_ROOT_DIR}/image-level'
        self.PLAYER_LEVEL_DIR = f'{self.FEATURES_ROOT_DIR}/player-level'

        self.LABELS = data['labels']
        self.TRAIN_IDS = data['train_ids']
        self.VAL_IDS = data['val_ids']
        self.TEST_IDS = data['test_ids']

        self._create_dirs()

    def _create_dirs(self):
        dirs = [self.FEATURES_ROOT_DIR,self.IMAGE_LEVEL_DIR, self.PLAYER_LEVEL_DIR ,self.ANNOT_SAVE_DIR]
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)


if __name__ == '__main__':
    config = CONFIG()
    train_len = len(config.TRAIN_IDS)
    val_len = len(config.VAL_IDS)
    test_len = len(config.TEST_IDS)

    total = train_len + val_len + test_len

    print(f'Train: {train_len}   -  {(train_len/total)*100:.1f}%     - if 80% train = {0.8 * total}')
    print(f'Val: {val_len}   -  {(val_len/total)*100:.1f}%   -  if 15%  test = {0.15 * total}')
    print(f'Test: {test_len}   -  {(test_len/total)*100:.1f}%   - if 5%  test = {0.05 * total}')