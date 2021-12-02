import argparse
import os

from tools.utils import display_summary_model
from kerascls.config import ConfigReader
from kerascls.model import KerasModel

if __name__ == '__main__':
    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config path',
                        deafult=os.path.join(package_dir, "configs", "setting.cfg"))

    print(os.path.join(package_dir, "configs", "setting.cfg"))

    config_reader = ConfigReader(parser.parse_args().config)
    model_config = config_reader.get_model_config()
    model_generator = KerasModel(**model_config, num_class=10)

    model_generator.create_full_model()
    display_summary_model(model_generator.full_model)
