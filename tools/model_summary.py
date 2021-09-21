import os
import argparse

from utils.config import get_model_from_config
from utils.model import KerasModel

if __name__ == '__main__':
    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config Path')
    parser.set_defaults(config=os.path.join(package_dir, "config/model.cfg"))

    model_config = get_model_from_config(parser.parse_args().config)
    model_generator = KerasModel(**model_config, num_class=10)

    model = model_generator.create_model_keras()
    print(model.summary())