#Use this file to get all user inputs into argparser
import argparse


def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    #Add arguments for image directory, model,
    parser.add_argument('--arch', type=str, default='resnet152', help='Set the CNN model')
    parser.add_argument('--GPU', default=False, help='Sets device to GPU or CPU')
    parser.add_argument('--topk', default=5, help='Returns number of predictions')
    parser.add_argument('--epochs', default=10, help='Set number of epochs to train the network')
    parser.add_argument('--learning_rate', default=.001, help='Set learning rate for model')
    parser.add_argument('--hidden_layers', type=int, default=1024, help='Set number of hidden layers')

    return parser.parse_args()