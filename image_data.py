import torch
from torchvision import datasets, transforms

def dataloader():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'


    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomRotation(5),
                                           transforms.RandomCrop(224),
                                           transforms.RandomHorizontalFlip(.05),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    train_image_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    validate_image_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_image_dataset, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(validate_image_dataset, batch_size=32)

    return trainloader, validloader, train_image_dataset, validate_image_dataset