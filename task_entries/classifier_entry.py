import torch
import wandb
from utils.tools import load_and_merge_config
from utils.registry import registry
from trainers.cnn_trainer import CNN4EMGTrainer
from trainers.mlp_classifier_trainer import DownstreamClassifierTrainer


@registry.register_task("classifier")
def run_classifier(args):
    args = load_and_merge_config(args)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"device: {device}")
    trainer = registry.get_trainer_class(args.trainer)(args)

    wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))

    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.exp_name))
    trainer.train()

    torch.cuda.empty_cache()
    wandb.finish()
