import wandb


class Monitoring:
    def __init__(self, project_name, correct_formula, x_range, experiment_config):
        wandb.init(project=project_name)
        table = wandb.Table(columns=['correct formula', 'x_range'])
        table.add_data(correct_formula, x_range)
        wandb.log({'formula settings': table})
        table = wandb.Table(columns=[*sorted(experiment_config.keys())])
        table.add_data(*[experiment_config[k] for k in sorted(experiment_config.keys())])
        wandb.log({'model config': table})

    def log(self, wandb_log):
        wandb.log(wandb_log)
