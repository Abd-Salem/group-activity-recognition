import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

class ModelB1(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(in_features=2048, out_features=1024, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                nn.Linear(in_features=1024, out_features=256, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Dropout(dropout),
                                nn.Linear(in_features=256, out_features=8, bias=True))

    def forward(self, x):
        out = self.classifier(x)
        return out



def train(model, dataset, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in loader:
            output = model(x_batch)
            loss = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch/{epoch+1}     -      batch-loss/{loss.item()}')






