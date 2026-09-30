from group_activity_recognition.trainer import NonTemporalTrainer
from group_activity_recognition.dataset_utils.volleyball_builders import build_loader
from group_activity_recognition.baselines.backbones import NonTemporalBackbone
from group_activity_recognition.helper_utils.more_helpers import get_config
import torch.nn as nn
import math, pytest





@pytest.fixture(scope="module")
def config():
    return get_config()



@pytest.fixture(scope="module")
def train_loader(config):
    return build_loader(config, 'train', shuffle=True)


@pytest.fixture(scope="module")
def val_loader(config):
    return build_loader(config, 'val', shuffle=False)


def test_trainer(config, train_loader, val_loader):
    """Smoke test: one epoch of NonTemporalTrainer, check outputs are sane."""
    device = config.get_device()
    model = NonTemporalBackbone(image_level=True).to(device=device)
    optimizer = config.OPTIMS['adamw'](model.parameters(), lr=config.LR[0])
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

    loss, acc = trainer.train(epochs=config.N_EPOCH[0], test_case=True)

    assert math.isfinite(loss)
    assert 0.0 <= acc <= 1.0
    assert (config.TEST_DIR / 'test.pt').exists()