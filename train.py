#imports
import torch
from torch import nn, optim
from torchvision import models
from train_args import get_train_args
from image_data import dataloader

# Load arguments from argparse into file
args = get_train_args()

#Set GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() and args.GPU else 'cpu')
# Load the model
model = eval('models.{}(pretrained=True)'.format(args.arch))


#Freeze Parameters
for param in model.parameters():
    param.requires_grad = False

if args.arch == 'resnet152':
    model = models.resnet152(pretrained=True)
    input_size = 2048
elif args.arch == 'densenet121':
    model = models.densenet121(pretrained=True)
    input_size = 1024

#Create new Classifier
classifier = nn.Sequential(nn.Linear(input_size, int(args.hidden_layers)),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(int(args.hidden_layers), 102),
                           nn.LogSoftmax(dim=1))

if model.fc:
    model.fc = classifier
else:
    model.classifier = classifier

#Create Loss and Optimizer
criterion = nn.NLLLoss()
if model.fc:
    optimizer = optim.Adam(model.fc.parameters(), lr=float(args.learning_rate))
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))


model.to(device)

#Training the Network
data = dataloader(args.dir)

#Initial Variables
epochs = args.epochs
running_loss = 0

for e in range(int(epochs)):
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
              f"Train loss: {running_loss/len(data[0]):.3f}.. "
              f"Validation loss: {validation_loss/len(data[1]):.3f}.. "
              f"Validation accuracy: {accuracy/len(data[1]):.3f}")
        running_loss = 0
        model.train()

def save_checkpoint(model, dataset, classifier, arch, optimizer):

    model.class_to_idx = dataset.class_to_idx
    checkpoint = {
                  'classifier': classifier,
                  'optimizer': optimizer,
                  'arch': arch,
                  'state_dict': model.state_dict(),
                  'optimizer_state': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx
                  }
    torch.save(checkpoint, 'checkpoints/checkpoint.pth')

save_checkpoint(model, data[2], classifier, args.arch, optimizer)