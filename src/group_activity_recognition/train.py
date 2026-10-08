import mlflow
import torch.nn as nn
from group_activity_recognition.trainer import NonTemporalTrainer
from group_activity_recognition.dataset_utils.volleyball_builders import build_loader
from group_activity_recognition.baselines.backbones import NonTemporalBackbone
from group_activity_recognition.helper_utils.mlflow_utils import start_run
from group_activity_recognition.helper_utils.more_helpers import get_config


def train_with_mlflow():
    config = get_config()
    print('Training Non Temp Backbone model')
    with start_run(tracking_uri=config.TRACKING_URI, experiment_name='Backbone', run_name='Non_tmp_backbone'):
        params = {'lr': config.LR[0], 'n_epochs': config.N_EPOCHS[1]}
        mlflow.log_params(params)
        mlflow.set_tags(
            {
                'model_part' : 'Backbone',
                'model_type' : 'Non-Temporal',
                'dataset'    : 'version 1.0',
                'stage'      : 'experiment'
            }
        )
        device = config.get_device()
        model = NonTemporalBackbone(image_level=True).to(device=device)
        optimizer = config.OPTIMS['adamw'](model.parameters(), lr=params['lr'])
        criterion = nn.CrossEntropyLoss()

        train_loader = build_loader(config=config, split_name='train', shuffle=True)
        val_loader = build_loader(config=config, split_name='val', shuffle=False)

        trainer = NonTemporalTrainer(
            model=model,
            train_loader=train_loader,
            validate_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint_dir=config.CHECKPOINT_DIR,
            model_name='Non-Tmp-Backbone'
        )
        loss, acc = trainer.train(epochs=params['n_epochs'])

        mlflow.log_metrics({'final_loss': loss, 'final_acc': acc})
        run = mlflow.active_run()
        print(f"{mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")


if __name__ == '__main__':
    train_with_mlflow()