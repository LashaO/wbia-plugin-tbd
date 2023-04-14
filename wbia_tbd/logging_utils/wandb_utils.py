import wandb
import os


def init_wandb(exp_name, project_name, config=None):

    print('Initializing wandb run')
    print('exp_name:', exp_name)
    print('project_name:', project_name)

    export_config = {k:v for k,v in dict(vars(config)).items() if not k.startswith('__')} if config else None


    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_API_KEY"] = "a7b8fd4b9c4346e595e83c9bec00b807f133d853"
    run = wandb.init(project=project_name, name=exp_name, config=export_config)
    # wandb.config = config  # {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

    return run
