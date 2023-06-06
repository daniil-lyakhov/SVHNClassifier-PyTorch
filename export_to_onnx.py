import torch
import argparse
from utils import get_model

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str, help='path to evaluate checkpoint, e.g. ./logs/model-100.pth')
parser.add_argument('-m', '--model_name', default='custom', type=str, help='Model name to use. Default custom')
parser.add_argument('-o', '--output_name', default='model.onnx', type=str, help='Onnx model name.')


def main(args):
    path_to_checkpoint_file = args.checkpoint
    model = get_model(args.model_name)
    model.restore(path_to_checkpoint_file)
    model.eval()
    inputs = torch.empty((1, 3, 54, 54))
    torch.onnx.export(model.float(), inputs, args.output_name)
    print('Done')


if __name__ == '__main__':
    main(parser.parse_args())
