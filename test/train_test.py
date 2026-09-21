from src.train import NonTemporalTrainer
from dataset_utils.volleyball_datasets import ImageLevelDataset
from helper_utils.feature_extraction import get_processor
from dataset_utils.volleyball_builders import load_clips_and_labels
from torch.utils.data import DataLoader
import torch.nn as nn
import math, pytest
from baselines.backbones import NonTemporalBackbone
from configs import CONFIG





@pytest.fixture(scope="module")
def config():
    return CONFIG()


def _build_loader(config, split_name, ids, shuffle):
    """Build an image-level DataLoader for the given split."""
    processor = get_processor(full_image=True, split=split_name)
    clips, labels, _ = load_clips_and_labels(split=ids, image_level=True, config=config)
    dataset = ImageLevelDataset(paths=clips, labels=labels, processor=processor, temporal=False)
    return DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE[1], shuffle=shuffle)


@pytest.fixture(scope="module")
def train_loader(config):
    return _build_loader(config, 'train', config.TRAIN_IDS, shuffle=True)


@pytest.fixture(scope="module")
def val_loader(config):
    return _build_loader(config, 'val', config.VAL_IDS, shuffle=False)


def test_trainer(config, train_loader, val_loader):
    """Smoke test: one epoch of NonTemporalTrainer, check outputs are sane."""
    device = config.get_device()
    model = NonTemporalBackbone(image_level=True).to(device=device)
    optimizer = config.OPTIMS['adamw'](model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    trainer = NonTemporalTrainer(
        model=model,
        train_loader=train_loader,
        validate_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=config.TEST_DIR,
        model_name='test'
    )

    loss, acc = trainer.train(epochs=1, test_case=True)

    assert math.isfinite(loss)
    assert 0.0 <= acc <= 1.0
    assert (config.TEST_DIR / 'test.pt').exists()