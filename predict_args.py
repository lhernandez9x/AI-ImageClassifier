#Use this file to get all user inputs into argparser
import argparse


def get_predict_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    #Add arguments for image directory, model,
    parser.add_argument('--label_map', type=bool, default=False, help='Sets device to GPU or CPU')
    parser.add_argument('--checkpoint', type=int, default=5, help='Returns number of predictions')
    parser.add_argument('--predict_image', type=int, default=10, help='Set number of epochs to train the network')
    parser.add_argument('--topk', type=int, default=5, help='Returns number of predictions')

    return parser.parse_args()