from trainers.AutoTimes_trainer import AutoTimesTrainer
from utils.tools import load_and_merge_config
from utils.registry import registry
import wandb
import torch


@registry.register_task("pretrain")
def run_pretrain(args):
    # Merge args and configs from yaml files
    args = load_and_merge_config(args)
    trainer = registry.get_trainer_class(args.trainer)(args)
    setting = '{}_{}_{}_epochs{}_sl{}_ll{}_tl{}_lr{}_bs{}_hd{}_hl{}_mix{}'.format(
        args.exp_name_prefix,
        args.feature_learner,
        args.data,
        args.train_epochs,
        args.seq_len,
        args.label_len,
        args.token_len,
        args.learning_rate,
        args.batch_size,
        args.mlp_hidden_dim,
        args.mlp_hidden_layers,
        args.mix_embeds,
    )

    wandb.init(
        project=args.wandb_project,
        name=setting,
        config=vars(args)
    )

    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    trainer.train(setting)

    torch.cuda.empty_cache()
    wandb.finish()
