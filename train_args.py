#Use this file to get all user inputs into argparser
import argparse


def get_train_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    #Add arguments for image directory, model,
    parser.add_argument('--dir', type=str, default='flowers', help='Path to directory')
    parser.add_argument('--arch', type=str, default='resnet152', help='Use Resnet152 or Densenet121 for models')
    parser.add_argument('--GPU', action='store_true', default=False, help='Use this to turn on the GPU')
    parser.add_argument('--epochs', type=int, default=10, help='Set number of epochs to train the network')
    parser.add_argument('--learning_rate', type=float, default=.001, help='Set learning rate for model')
    parser.add_argument('--hidden_layers', type=int, default=1024, help='Set number of hidden layers')

    return parser.parse_args()

args = get_train_args()
print(args.dir)
print(args.arch)
print(args.GPU)
print(args.epochs)
print(args.learning_rate)
print(args.hidden_layers)
