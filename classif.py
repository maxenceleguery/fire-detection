import torch
import torch.nn as nn
from dataset import get_dataloaders

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() # why is this necessary?
        self.network = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=2, stride=1, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(in_features=61952, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=2), # last unit has 
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    train_load, test_load, val_load = get_dataloaders()
    model = CNN().cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    num_epochs = 10

    for epoch in range(num_epochs):
        total = 0
        wrong = 0
        for i, (images, labels) in enumerate(train_load):
            images, labels = images.cuda(), labels.cuda()
            out = model(images)

            predictions = torch.argmax(out, dim=1)
            incorrect_indices = (predictions.squeeze() != labels).nonzero().squeeze()

            total += predictions.shape[0]
            wrong += incorrect_indices.shape[0]

            optimizer.zero_grad()
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if i%5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Iter [{i+1}/] Train loss : {loss.item():.3f}")
        print(f"Epoch [{epoch+1}/{num_epochs}] Train acc : {100 - 100.*wrong/total:.2f}")

        with torch.no_grad():
            total = 0
            wrong = 0
            for i, (images, labels) in enumerate(val_load):
                images, labels = images.cuda(), labels.cuda()
                out = model(images)

                predictions = torch.argmax(out, dim=1)
                incorrect_indices = (predictions.squeeze() != labels).nonzero().squeeze()

                total += predictions.shape[0]
                wrong += incorrect_indices.shape[0]
            print(f"Epoch [{epoch+1}/{num_epochs}] Val acc : {100 - 100.*wrong/total:.2f}")