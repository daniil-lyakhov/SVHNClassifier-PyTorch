import os
import torch
from torch import nn
import glob

class BaseModelMixIn:
    def _get_step(self, path_to_model):
        return path_to_model.split('/')[-1].split('-')[-1].split('.')[0]

    def store(self, path_to_dir, step, maximum=5):
        file_name = self.__class__.__name__ + '-{}.pth'
        path_to_models = glob.glob(os.path.join(path_to_dir, file_name.format('*')))
        if len(path_to_models) == maximum:
            min_step = min([int(self._get_step(path_to_model)) for path_to_model in path_to_models])
            path_to_min_step_model = os.path.join(path_to_dir, file_name.format(min_step))
            os.remove(path_to_min_step_model)

        path_to_checkpoint_file = os.path.join(path_to_dir, file_name.format(step))
        torch.save(self.state_dict(), path_to_checkpoint_file)
        return path_to_checkpoint_file

    def restore(self, path_to_checkpoint_file):
        self.load_state_dict(torch.load(path_to_checkpoint_file))
        step = int(self._get_step(path_to_checkpoint_file))
        return step


class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_digits = 5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._digit_length = nn.Linear(in_features, 7)
        self._digits = nn.ModuleList([nn.Linear(in_features, 11) for _ in range(num_digits)])

    def forward(self, x):
        output = [self._digit_length(x)]
        for module in self._digits:
            output.append(module(x))
        return output
