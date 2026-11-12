import argparse
import yaml
from lib.util import get_logger_simple
from train import Runner
from datetime import datetime
import os


def main(args):
    with open(args.config_filename) as f:
        config = yaml.safe_load(f)
        config['log_dir'] = log_dir
        config_save_path = os.path.join(log_dir, "config.yml")
        yaml.dump(config, open(config_save_path, "w"))

        config['logger'] = logger
        runner = Runner(**config)

        if config['mode'] == 'train':
            runner.train()
        # if config['mode'] == 'test':
        #     runner.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Baselines - NeRF2')
    parser.add_argument('--exp_name', default='rfid-spectrum', help='experiment name', choices=['mimo-csi', 'ble-rssi', 'rfid-spectrum'])
    parser.add_argument('--load_checkpoint', default=False, help='load the checkpoint')
    parser.add_argument('--run_name', type=str, default='mimo-csi_20251204_1116', help='Run directory name under logs/, e.g. "mimo-csi_20251203_0901" when loading checkpoint.'
    )
    args = parser.parse_args()

    # ===== ① Load configuration files =====
    if args.load_checkpoint:
        if args.run_name is None:
            raise ValueError("When --load_checkpoint is True, you must provide --run_name, "
                             "e.g. --run_name mimo-csi_20251203_0901")
        run_name = args.run_name
        log_dir = os.path.join("logs", run_name)  # logs/mimo-csi_20251203_0901
        args.config_filename = os.path.join(log_dir, "config.yml")
        if not os.path.exists(args.config_filename):
            raise FileNotFoundError(f"Config file not found: {args.config_filename}")
    else:
        args.config_filename = f'./configs/{args.exp_name}.yml'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"{args.exp_name}_{timestamp}"
        log_dir = os.path.join("logs", run_name)

    # ===== ② Create logger =====
    logger = get_logger_simple(log_dir, f'exp_{args.exp_name}')
    logger.info(f"Running experiment: {args.exp_name}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Config file: {args.config_filename}")
    logger.info(f"Load checkpoint: {args.load_checkpoint}")


    main(args)