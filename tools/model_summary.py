import os
import argparse

from utils.config import ConfigReader
from utils.model import KerasModel

if __name__ == '__main__':
    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config path')
    parser.set_defaults(config=os.path.join(package_dir, "configs", "setting.cfg"))

    config_reader = ConfigReader(parser.parse_args().config)
    model_config = config_reader.get_model()
    model_generator = KerasModel(**model_config, num_class=10)

    model = model_generator.create_model_keras()
    print(model.summary())