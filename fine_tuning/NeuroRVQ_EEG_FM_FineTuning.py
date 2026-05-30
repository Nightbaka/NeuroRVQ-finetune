import numpy as np
import torch
from lightning.pytorch import LightningModule, Trainer
from inference.modules.NeuroRVQ_EEG_tokenizer_inference_modules import ch_names_global, create_embedding_ix, check_model_eval_mode
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader
import warnings

def get_class_weights(y, n_cls):
    y = torch.Tensor(y)
    class_weights = torch.unique(y, return_counts=True)[1]
    class_weights = 1 / class_weights
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights * len(torch.unique(y))  # (n_classes,)
    if len(class_weights) < n_cls:
        tmp = class_weights
        class_weights = torch.zeros(n_cls)
        class_weights[:len(tmp)] = tmp
    class_weights = class_weights.cuda()
    return class_weights

class NeuroRVQModule(LightningModule):
    '''
    Module that performs fine-tuning of NeuroRVQ
    '''
    def __init__(self, sample_length, chnames, n_out, train_head_only, args, foundation_model):
        super().__init__()
        self.save_hyperparameters(ignore=['foundation_model'])
        self.n_time = sample_length // args['patch_size']
        chnames = np.array([c.lower().encode() for c in chnames])
        self.chmask = np.isin(chnames, ch_names_global)
        self.chnames = chnames[self.chmask]
        self.n_out = n_out
        self.model = foundation_model
        self.train_head_only = train_head_only
        self.d_out = self.n_out if self.n_out > 2 else 1
        self.model.reset_classifier(self.d_out)
        self.criterion = F.cross_entropy if self.n_out > 2 else F.binary_cross_entropy_with_logits
        self.results = {'train_accuracy': [], 'val_accuracy': [], 'train_bacc': [], 'val_bacc': []}
        self.weight_decay = args['weight_decay_finetuning']
        self.warmup_epochs = args['warmup_epochs_finetuning']
        self.amp_dtype = torch.bfloat16
        self.lr = float(args['lr_finetuning'])
        self.layer_decay = float(args['layer_decay_finetuning'])
        self.n_patches = args['n_patches']
        self.patch_size = args['patch_size']
        self.temp_embed_ix = None
        self.spat_embed_ix = None
        self.class_weights = None
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []

        if self.train_head_only:
            for name, param in self.model.named_parameters():
                if 'head.' in name or 'fc_norm.' in name:
                    continue
                param.requires_grad = False

    def setup(self, stage=None):
        if self.temp_embed_ix is None or self.spat_embed_ix is None:
            temp_embed_ix, spat_embed_ix = create_embedding_ix(
                self.n_time,
                self.n_patches,
                self.chnames,
                ch_names_global,
            )
            self.temp_embed_ix = temp_embed_ix
            self.spat_embed_ix = spat_embed_ix

    def size(self):
        """ Returns number of trainable parameters in model """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def forward(self, x):
        temp_embed_ix = self.temp_embed_ix.to(self.device)
        spat_embed_ix = self.spat_embed_ix.to(self.device)
        return self.model(x, temp_embed_ix, spat_embed_ix)

    def _build_param_groups(self):
        param_groups = {}
        for i_m, (p_name, param) in enumerate(self.model.named_parameters()):  # model layers
            if not param.requires_grad:
                continue
            if ('head.' in p_name) or ('fc_norm.' in p_name):  # normal lr for classification head
                param_groups[p_name] = {'params': [param],
                                        'weight_decay': self.weight_decay,
                                        'lr': self.lr}
            else:
                param_groups[p_name] = {'params': [param],
                                        'weight_decay': self.weight_decay,
                                        'lr': self.lr * self.layer_decay ** (
                                                len(list(self.model.named_parameters())) - i_m)}
        return list(param_groups.values())

    def _shared_step(self, batch, stage):
        x_b, y_b = batch
        x_b = x_b[:, self.chmask, :]
        n, c, t = x_b.shape
        x_b = x_b.reshape(n, c, self.n_time, self.patch_size)
        y_b = y_b.long() if self.n_out > 2 else y_b.float()

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.device.type == 'cuda'):
            p, _ = self(x_b)
            p = p.squeeze(-1)
            loss_weight = self.class_weights.to(self.device) if p.ndim == 2 else self.class_weights.to(self.device)[y_b.long()]
            loss = self.criterion(p, y_b.to(self.device), weight=loss_weight)

        pred = p.detach().cpu().float()
        pred = pred.argmax(dim=-1) if pred.ndim == 2 else torch.round(torch.sigmoid(pred))
        target = y_b.detach().cpu()

        if stage == 'train':
            self.train_preds.append(pred.numpy())
            self.train_targets.append(target.numpy())
        else:
            self.val_preds.append(pred.numpy())
            self.val_targets.append(target.numpy())

        self.log(f'{stage}_loss', loss, prog_bar=(stage == 'val'), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, 'val')

    def on_train_epoch_start(self):
        self.train_preds = []
        self.train_targets = []

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_targets = []

    def on_train_epoch_end(self):
        if self.train_preds:
            y_pred = np.concatenate(self.train_preds)
            y_true = np.concatenate(self.train_targets)
            train_acc = accuracy_score(y_true, y_pred)
            train_bacc = balanced_accuracy_score(y_true, y_pred)
            self.results['train_accuracy'].append(train_acc)
            self.results['train_bacc'].append(train_bacc)
            self.log('train_accuracy', train_acc, prog_bar=False)
            self.log('train_bacc', train_bacc, prog_bar=False)

    def on_validation_epoch_end(self):
        if self.val_preds:
            y_pred = np.concatenate(self.val_preds)
            y_true = np.concatenate(self.val_targets)
            val_acc = accuracy_score(y_true, y_pred)
            val_bacc = balanced_accuracy_score(y_true, y_pred)
            self.results['val_accuracy'].append(val_acc)
            self.results['val_bacc'].append(val_bacc)
            self.log('val_accuracy', val_acc, prog_bar=True)
            self.log('val_bacc', val_bacc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self._build_param_groups())

        steps_per_epoch = max(1, getattr(self.trainer, 'num_training_batches', 1))
        total_epochs = max(1, getattr(self.trainer, 'max_epochs', 1))

        if total_epochs < self.warmup_epochs + 1:
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-1,
                end_factor=1,
                total_iters=total_epochs * steps_per_epoch,
            )
        else:
            scheduler1 = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-1,
                end_factor=1,
                total_iters=self.warmup_epochs * steps_per_epoch,
            )
            scheduler2 = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1,
                end_factor=1e-1,
                total_iters=(total_epochs - self.warmup_epochs) * steps_per_epoch,
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                [scheduler1, scheduler2],
                milestones=[self.warmup_epochs * steps_per_epoch],
            )

        warnings.filterwarnings('ignore', category=UserWarning, module='torch.optim.lr_scheduler')
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
            },
        }

    def fit(self, train_dataset, validation_dataset, batch_size, epochs):
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []

        y_train = [ys for _, ys in train_dataset]
        y_val = [ys for _, ys in validation_dataset]
        y = y_train + y_val
        self.class_weights = get_class_weights(y, self.n_out)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        trainer = Trainer(
            max_epochs=epochs,
            accelerator='auto',
            devices=1,
            log_every_n_steps=1,
        )
        trainer.fit(self, train_dataloader, val_dataloader)
