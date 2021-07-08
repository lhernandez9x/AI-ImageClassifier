#Use this file to get all user inputs into argparser
import argparse


def get_predict_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    #Add arguments for image directory, model,
    parser.add_argument('--label_map', type=str, default='cat_to_name.json', help='Path to label file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint.pth', help='Path to checkpoint file')
    parser.add_argument('--predict_image', type=str, default='flowers/test/60/image_02932.jpg',
                        help='Path for image to run in predict')
    parser.add_argument('--topk', type=int, default=5, help='Returns number of predictions')

    return parser.parse_args()