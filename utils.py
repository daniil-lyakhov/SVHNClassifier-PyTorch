from model import Model
from modeling.mobilenet_v3_small import get_mobilenet_v3_small
from modeling.mobilenet_v3_small import get_mobilenet_v3_large


def get_model(model_name: str):
    if  model_name == 'custom':
        return Model()
    if model_name == 'mobilenet_v3_small':
       return get_mobilenet_v3_small()
    if model_name == 'mobilenet_v3_large':
        return get_mobilenet_v3_large()
    raise RuntimeError(f'Wrong model name: {model_name}')
