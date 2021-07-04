#imports
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from get_args import get_input_args

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

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=float(args.learning_rate))

model.to(device)

