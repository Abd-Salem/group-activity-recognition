from src.train import NonTemporalTrainer
from dataset_utils.volleyball_datasets import ImageLevelDataset
from helper_utils.feature_extraction import get_processor
from dataset_utils.volleyball_builders import load_clips_and_labels
from torch.utils.data import DataLoader
import torch.nn as nn
import torch, Path
from baselines.backbones import NonTemporalBackbone
from configs import CONFIG


def test_trainer():
    config = CONFIG()

    # trian split
    processor = get_processor(full_image=True, split='train')
    train_clips, train_labels, _ = load_clips_and_labels(split=config.TRAIN_IDS, image_level=True, config=config)
    train_dataset = ImageLevelDataset(paths=train_clips, labels=train_labels, processor=processor, temporal=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE[1], shuffle=True)

    # val split
    processor = get_processor(full_image=True, split='val')
    val_clips, val_labels, _ = load_clips_and_labels(split=config.VAL_IDS, image_level=True, config=config)
    val_dataset = ImageLevelDataset(paths=val_clips, labels=val_labels, processor=processor, temporal=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.BATCH_SIZE[1], shuffle=True)

    model = NonTemporalBackbone(image_level=True)
    optimizer = config.OPTIMS['adamw'](model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    ckpt = Path(config.ROOT) / 'test'

    trainer = NonTemporalTrainer(
        model=model,
        train_loader=train_loader,
        validate_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=config.get_device(),
        checkpoint_dir= ckpt,
        model_name='test'
        )

    loss, acc = trainer.train(epochs=1, test_case=True)


