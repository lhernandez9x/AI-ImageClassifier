#imports
import torch
from torch import nn, optim
from torchvision import models
from get_args import get_input_args
from image_data import dataloader

# Load arguments from argparse into file
args = get_input_args()

#Set GPU or CPU
device = torch.device('cuda' if args.GPU else 'cpu')
# Load the model
model = eval('models.{}(pretrained=True)'.format(args.arch))


#Freeze Parameters
for param in model.parameters():
    param.requires_grad = False

#Create new Classifier
classifier = nn.Sequential(nn.Linear(2048, int(args.hidden_layers)),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(int(args.hidden_layers), 102),
                           nn.LogSoftmax(dim=1))

if model.fc:
    model.fc = classifier
else:
    model.classifier = classifier
#Create Loss and Optimizer

print(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=float(args.learning_rate))
print(optimizer)

model.to(device)

#Training the Network
data = dataloader()

#Initial Variables
epochs = args.epochs
print(epochs)
running_loss = 0

for e in range(epochs):
    for inputs, labels in data[0]:
        inputs, labels = inputs.to(device), labels.to(device)

        #reset Optimizer
        optimizer.zero_grad()

        log_probs = model.forward(inputs)
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    validation_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in data[1]:
            inputs, labels = inputs.to(device), labels.to(device)

            validps = model.forward(inputs)
            running_validation_loss = criterion(validps, labels)

            validation_loss += running_validation_loss.item()

            # Calculate accuracy
            ps = torch.exp(validps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {e+1}/{epochs}.. "
              f"Train loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation loss: {validation_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader):.3f}")
        running_loss = 0
        model.train()