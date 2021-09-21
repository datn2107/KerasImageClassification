import os
import argparse

from utils.config import get_optimizer_from_config, get_loss_from_config, get_list_metric_from_config
from utils.prepare_compiler import load_optimizer, load_loss, load_list_metric

if __name__ == '__main__':
    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config Path')
    parser.set_defaults(config=os.path.join(package_dir, "config/training.cfg"))

    optimizer_info = get_optimizer_from_config(parser.parse_args().config)
    loss_info = get_loss_from_config(parser.parse_args().config)
    list_metric_info = get_list_metric_from_config(parser.parse_args().config)

    print("\nOptimizer")
    print(load_optimizer(**optimizer_info).get_config())
    print()

    print("\nLoss")
    print(load_loss(**loss_info).get_config())
    print()

    list_metric = load_list_metric(list_metric_info)
    print("\nMetrics")
    for metric in list_metric:
        if metric != 'accuracy':
            print(metric.get_config())
    print()