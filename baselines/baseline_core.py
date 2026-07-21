import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from helper_utils.configs import CONFIG
from dataset_utils.baseline_dataset_loader_utils import ImageDataset
from helper_utils.helpers import get_processor
from sklearn.metrics import classification_report


'''
- The dataset is 55 videos. Each video has a directory for it with sequential IDs (0, 1...54)
	- Train Videos: 1 3 6 7 10 13 15 16 18 22 23 31 32 36 38 39 40 41 42 48 50 52 53 54
	- Validation Videos: 0 2 8 12 17 19 24 26 27 28 30 33 46 49 51
	- Test Videos: 4 5 9 11 14 20 21 25 29 34 35 37 43 44 45 47
	
	
	Group Activity Class 		No. of Instances
Right set 			644
Right spike 		623
Right pass 		801
Right winpoint 		295
Left winpoint 		367
Left pass 			826
Left spike 			642
Left set 			633


Action Classes 		No. of Instances
Waiting 			3601
Setting 			1332
Digging 			2333
Falling 			1241
Spiking 			1216
Blocking 			2458
Jumping 			341
Moving 			5121
Standing 			38696

'''


class BaseLineModel(nn.Module):
    '''
        generic class to create baselines
    '''
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)
        return out


class BL1_HEAD(nn.Module):
    def __init__(self, input_dim=2048, output_dim=8, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(in_features=input_dim, out_features=1024, bias=True),
                                nn.BatchNorm1d(num_features=1024),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                nn.Linear(in_features=1024, out_features=256, bias=True),
                                nn.BatchNorm1d(num_features=256),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                nn.Linear(in_features=256, out_features=output_dim, bias=True))

    def forward(self, x):
        out = self.classifier(x)
        return out


def train_baseline(model, dataset, batch_size=32, epochs=50, opt_lr=0.0001):
    '''
    training baselines script
    :param model: baseline model
    :param dataset: training dataset
    :param batch_size: 32, 64, 128, ...
    :param epochs: number of epochs
    :param opt_lr: optimizer learning rate
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=opt_lr
    )

    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch/{epoch+1}     -      batch({idx})-loss/{loss.item()}')


@torch.no_grad()
def evaluate_baseline(model, dataset):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    total_loss = 0
    all_predictions, all_labels =  [], []
    total = len(dataset)

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)
        losses = criterion(logits, labels)
        total_loss += losses.item()

        all_predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        # correct += (labels == predictions).sum().item()

    avg_loss = total_loss / total
    all_predictions = np.concatenate(all_predictions)  # merge all batches into one flat array
    all_labels = np.concatenate(all_labels)

    return avg_loss, all_predictions, all_labels


if __name__ == '__main__':
    config = CONFIG()
    num_classes = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare backbone
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    print(resnet)
    input_dim = resnet.fc.in_features
    backbone_design = list(resnet.children())[:-1]           # drop fc layer
    backbone = torch.nn.Sequential(*backbone_design)

    # baseline 1
    baseline1_model = BaseLineModel(backbone=backbone, head=BL1_HEAD(input_dim=input_dim, output_dim=num_classes)).to(device)

    # prepare train dataset
    train_paths, train_labels = load_image_paths_labels(split=config.TRAIN_IDS)
    processor = get_processor(full_image=True)
    train_dataset = ImageDataset(image_paths=train_paths, labels=train_labels, processor=processor)

    # train baseline
    train_baseline(baseline1_model, train_dataset, batch_size=64, epochs=50, opt_lr=0.001)

    # prepare val dataset
    val_paths, val_labels = load_image_paths_labels(split=config.VAL_IDS)
    processor = get_processor(full_image=True)
    val_dataset = ImageDataset(image_paths=val_paths, labels=val_labels, processor=processor)

    # evaluate model and print classification report
    avg_loss, predictions, labels = evaluate_baseline(baseline1_model, val_dataset)
    print(f'Avg. Loss: {avg_loss}')
    print(classification_report(labels, predictions, target_names=list(config.LABELS.keys())))




