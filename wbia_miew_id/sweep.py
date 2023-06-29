import optuna
import yaml
from train import run
from helpers import get_config
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pickle


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to the YAML configuration file. Default: configs/default_config.yaml",
    )
    return parser.parse_args()


def objective(trial, config):

    # Specify the parameters you want to optimize
    config.data.train_n_filter_min = trial.suggest_int("train_n_filter_min", 2, 5)
    image_size = trial.suggest_categorical("image_size", [192, 256, 384, 440, 512])
    config.data.image_size = [image_size, image_size]
    n_epochs = trial.suggest_int("epochs", 20, 40)
    config.engine.epochs = n_epochs
    config.model_params.margin = trial.suggest_uniform("margin", 0.1, 0.7)
    config.model_params.s = trial.suggest_uniform("s", 20, 64)

    # The scheduler params are derived from one base paremeter to minimize the number of parameters to optimzie
    lr_base = trial.suggest_loguniform("lr_base", 1e-6, 1e-2)
    config.scheduler_params.lr_start = lr_base
    config.scheduler_params.lr_max = lr_base * 10
    config.scheduler_params.lr_min = lr_base / 2
    result = run(config)

    print("cfg", config.engine)

    return result


if __name__ == "__main__":
    #     args = parse_args()
    config_path = "configs/default_config.yaml"  # args.config

    config = get_config(config_path)

    study = optuna.create_study(
        sampler=TPESampler(), pruner=MedianPruner(), direction="maximize"
    )

    comb_objective = lambda trial: objective(trial, config)

    study.optimize(comb_objective, n_trials=100)

    print("Best trial:")
    trial_ = study.best_trial

    print(f"Value: {trial_.value}")

    print("Best parameters:")
    for key, value in trial_.params.items():
        print(f"    {key}: {value}")

    # saves best parameters
    save_dict = trial_.params
    save_dict['best_score'] = trial_.value

    with open('sweep.pkl', 'wb') as f:
        pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
