import os
import time
import torch
import warnings
import wandb
import numpy as np
import csv
from datetime import datetime
from trainers.base_pre_trainer import BasePreTrainer
from utils.registry import registry
from utils.tools import EarlyStopping, adjust_learning_rate
from data_provider.data_factory import data_provider
from torch import optim
import torch.nn as nn
from feature_learners.AutoTimes_Llama import AutoTimesLlama


warnings.filterwarnings('ignore')

@registry.register_trainer("AutoTimes")
class AutoTimesTrainer(BasePreTrainer):
    def __init__(self, args):
        super(AutoTimesTrainer, self).__init__(args)

    def _build_model(self):
        model = registry.get_feature_learner_class(self.args.feature_learner)(self.args).to(self.device)
        
        return model
    
    def _get_data(self):
        ssl_train_loader, ssl_val_loader = data_provider(self.args)

        return ssl_train_loader, ssl_val_loader

    def _get_timestamp_embedding(self):
        te_filename = f"sl{self.args.seq_len}_tl{self.args.token_len}_sr{self.args.sampling_rate}.pt"
        te_path = os.path.join(self.args.timestamp_embedding_path, te_filename)
        te = torch.load(te_path)

        return te

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                print(n, p.dtype, p.shape)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        print('next learning rate is {}'.format(self.args.learning_rate))

        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()

        return criterion

    def train(self, setting):
        ssl_train_loader, ssl_val_loader = self._get_data()
        te = self._get_timestamp_embedding()
        te = te.float().to(self.device)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        log_path = os.path.join("logs", setting)
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        log_file_path = os.path.join(log_path, "pretrain_log.csv")
        with open(log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "vali_loss", "epoch_timestamp"])

        time_now = time.time()

        train_steps = len(ssl_train_loader)
        print(f"{train_steps} steps in one epoch.")
        early_stopping = EarlyStopping(self.args, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            loss_val = torch.tensor(0., device=self.device)
            count = torch.tensor(0., device=self.device)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(ssl_train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, te)
                        loss = criterion(outputs, batch_y)
                        loss_val += loss
                        count += 1
                else:
                    outputs = self.model(batch_x, te)
                    loss = criterion(outputs, batch_y)
                    loss_val += loss
                    count += 1

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    wandb.log({'iter_train_loss': loss, 'iters': i+1})

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(ssl_val_loader, criterion)
            wandb.log({'epoch':epoch+1, 'train_loss':train_loss, 'vali_loss':vali_loss})
            with open(log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                epoch_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow([epoch+1, train_loss, vali_loss, epoch_timestamp])

            print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model
    
    def vali(self, vali_loader, criterion):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        te = self._get_timestamp_embedding()
        te = te.float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, te)
                else:
                    outputs = self.model(batch_x, te)

                outputs = outputs[:, :, :]
                batch_y = batch_y[:, :, :].to(self.device)
                loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (test_steps - i)
                    print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                    iter_count = 0
                    time_now = time.time()

        total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

