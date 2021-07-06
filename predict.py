#imports
import torch
import numpy as np
from get_args import get_input_args
from PIL import Image
import json

#load network
checkpoint = torch.load('checkpoints/checkpoint.pth')
#load args
args = get_input_args()

#use GPU or CPU
device = torch.device('cuda' if args.GPU else 'cpu')

def process_image(image):

    small_side = 256
    width, height = image.size

    # Resizing image to 256 on the smallest size
    if width > height:
        width = int(small_side * (width / height))
        height = small_side

    elif width < height:
        height = int(small_side * (width / height))
        width = small_side

    # Resized image
    image = image.resize((width, height))

    # Crop the image to 224 from the center
    crop_size = 224
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2

    # Cropped image
    image = image.crop((left, top, right, bottom))

    # Convert image to np array
    np_image = np.array(image)

    #
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # normalize
    np_image = np.interp(np_image, (0, 255), (0, 1))
    np_image -= means
    np_image /= std

    # arrange data for Pytorch and Numpy/PIL
    np_image = np_image.transpose((2, 0, 1))

    # final image
    image = torch.from_numpy(np_image)

    return image


def predict(image_path, model, topk):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    image = Image.open(image_path)
    image = process_image(image)

    model.to(device)
    model.eval()

    with torch.no_grad():
        # Add for single image use in model and converting
        image = image.to(device)
        image = image.unsqueeze_(0)
        image = image.float()

        output = model.forward(image)

        probs = torch.exp(output)
        probs = probs.topk(topk)

        # converts probabilities and classes into numpy arrays
        probabilities = probs[0].numpy()[0]
        classes = [str(x) for x in probs[1].numpy()[0]]

        # convert classes to flower name
        idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
        name = [cat_to_name[cls] for cls in classes]
        classes = name

    return probabilities, classes

probabilities, classes = predict('flowers/test/60/image_02932.jpg', checkpoint['model'], args.topk)

print(probabilities, classes)

