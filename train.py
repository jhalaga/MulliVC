"""
Training script for MulliVC.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import wandb
from tqdm import tqdm
import argparse
from typing import Dict, Any, Optional

from models.mullivc import MulliVC, create_mullivc_model
from utils.data_utils import create_dataloader, load_config
from utils.audio_utils import AudioProcessor
from utils.model_utils import get_runtime_device


class MulliVCTrainer:
    """Trainer for MulliVC."""
    
    def __init__(self, config_path: str, overrides: Optional[Dict[str, Any]] = None):
        self.config = load_config(config_path)
        self._apply_overrides(overrides or {})
        self.device = get_runtime_device()
        
        # Initialize the model
        self.model = create_mullivc_model(config_path).to(self.device)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            [
                {'params': self.model.content_encoder.parameters()},
                {'params': self.model.timbre_encoder.parameters()},
                {'params': self.model.fine_grained_conformer.parameters()},
                {'params': self.model.mel_decoder.parameters()}
            ],
            lr=self.config['training']['learning_rate'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2']),
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2']),
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g,
            T_max=self.config['training']['num_epochs']
        )
        
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d,
            T_max=self.config['training']['num_epochs']
        )
        
        # Audio processor
        self.audio_processor = AudioProcessor(self.config)
        
        # Initialize wandb
        if self.config.get('wandb', {}).get('enabled', False):
            wandb.init(
                project=self.config['wandb']['project'],
                name=self.config['wandb']['name'],
                config=self.config
            )

    def _apply_overrides(self, overrides: Dict[str, Any]):
        """Applies CLI overrides to the loaded configuration."""
        data_config = self.config.setdefault('data', {})
        training_config = self.config.setdefault('training', {})

        if overrides.get('batch_size') is not None:
            data_config['batch_size'] = overrides['batch_size']
        if overrides.get('num_workers') is not None:
            data_config['num_workers'] = overrides['num_workers']
        if overrides.get('max_train_samples') is not None:
            data_config['max_train_samples'] = overrides['max_train_samples']
        if overrides.get('max_val_samples') is not None:
            data_config['max_validation_samples'] = overrides['max_val_samples']
        if overrides.get('epochs') is not None:
            training_config['num_epochs'] = overrides['epochs']
        if overrides.get('steps_per_epoch') is not None:
            training_config['steps_per_epoch'] = overrides['steps_per_epoch']
        if overrides.get('validation_steps') is not None:
            training_config['validation_steps'] = overrides['validation_steps']
        if overrides.get('enable_cycle_consistency'):
            training_config['enable_cycle_consistency'] = True
        if overrides.get('disable_wandb'):
            wandb_config = self.config.setdefault('wandb', {})
            wandb_config['enabled'] = False

    def _bounded_num_batches(self, dataloader: DataLoader, limit: Optional[int]) -> int:
        """Returns the effective number of batches for this pass."""
        num_batches = len(dataloader)
        if limit is not None:
            num_batches = min(num_batches, int(limit))
        return num_batches
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Trains the model for one epoch."""
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'timbre': 0.0,
            'pitch': 0.0,
            'adversarial': 0.0,
            'asr': 0.0
        }
        
        num_batches = self._bounded_num_batches(
            dataloader,
            self.config['training'].get('steps_per_epoch')
        )
        if num_batches == 0:
            raise RuntimeError('No batches available for training.')

        with tqdm(total=num_batches, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                # Move data to the device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Train the generator
                self.optimizer_g.zero_grad()
                losses_g = self.model.training_step(batch, batch_idx)
                loss_g = losses_g['total']
                loss_g.backward()
                self.optimizer_g.step()
                
                # Train the discriminator
                self.optimizer_d.zero_grad()
                losses_d = self._train_discriminator(batch)
                loss_d = losses_d['total']
                loss_d.backward()
                self.optimizer_d.step()
                
                # Update total losses
                for key in total_losses:
                    if key in losses_g:
                        total_losses[key] += losses_g[key].item()
                    if key in losses_d:
                        total_losses[key] += losses_d[key].item()
                
                # Update the progress bar
                pbar.set_postfix({
                    'G_Loss': f'{loss_g.item():.4f}',
                    'D_Loss': f'{loss_d.item():.4f}'
                })
                pbar.update(1)
                
                # Logging
                if batch_idx % self.config['training']['log_interval'] == 0:
                    self._log_metrics(epoch, batch_idx, losses_g, losses_d)
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches
        
        return total_losses
    
    def _train_discriminator(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Trains the discriminator."""
        source_audio = batch['audio']
        target_audio = batch['audio']  # Same audio for discriminator training
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model.forward(source_audio, target_audio)
            generated_mel = outputs['generated_mel']
        
        # Real samples
        real_mel = self.audio_processor.audio_to_mel(target_audio)
        real_disc_output, _ = self.model.discriminator(real_mel)
        
        # Fake samples
        fake_disc_output, _ = self.model.discriminator(generated_mel)
        
        # Discriminator loss
        real_loss = self.model.loss_fn.adversarial_loss(real_disc_output, is_real=True)
        fake_loss = self.model.loss_fn.adversarial_loss(fake_disc_output, is_real=False)
        
        discriminator_loss = (real_loss + fake_loss) / 2
        
        return {
            'total': discriminator_loss,
            'adversarial': discriminator_loss
        }
    
    def _log_metrics(self, epoch: int, batch_idx: int, losses_g: Dict, losses_d: Dict):
        """Logs metrics."""
        if self.config.get('wandb', {}).get('enabled', False):
            metrics = {
                'epoch': epoch,
                'batch': batch_idx,
                'generator/total_loss': losses_g['total'].item(),
                'generator/reconstruction_loss': losses_g['reconstruction'].item(),
                'generator/timbre_loss': losses_g['timbre'].item(),
                'generator/pitch_loss': losses_g['pitch'].item(),
                'generator/asr_loss': losses_g['asr'].item(),
                'discriminator/total_loss': losses_d['total'].item(),
                'discriminator/adversarial_loss': losses_d['adversarial'].item()
            }
            wandb.log(metrics)
    
    def validate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validates the model."""
        self.model.eval()
        
        total_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'timbre': 0.0,
            'pitch': 0.0,
            'adversarial': 0.0,
            'asr': 0.0
        }
        
        num_batches = self._bounded_num_batches(
            dataloader,
            self.config['training'].get('validation_steps')
        )
        if num_batches == 0:
            raise RuntimeError('No batches available for validation.')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc='Validation')):
                if batch_idx >= num_batches:
                    break

                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model.forward(batch['audio'], batch['audio'])
                
                # Targets
                targets = {
                    'target_mel': self.audio_processor.audio_to_mel(batch['audio']),
                    'target_timbre': outputs['timbre_features']
                }
                
                # Compute losses
                losses = self.model.compute_losses(outputs, targets, is_real=True)
                
                # Accumulate losses
                for key in total_losses:
                    if key in losses:
                        total_losses[key] += losses[key].item()
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches
        
        return total_losses
    
    def save_checkpoint(self, epoch: int, step: int, is_best: bool = False):
        """Saves a checkpoint."""
        checkpoint_dir = self.config['paths']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_step_{step}.pt')
        self.model.save_checkpoint(checkpoint_path, epoch, step)
        
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            self.model.save_checkpoint(best_path, epoch, step)
    
    def train(self):
        """Main training loop."""
        # Create dataloaders
        train_dataloader = create_dataloader(self.config, split='train')
        val_dataloader = create_dataloader(self.config, split='validation', shuffle=False)

        print(
            f"Device: {self.device} | batch_size={self.config['data']['batch_size']} | "
            f"max_train_samples={self.config['data'].get('max_train_samples')} | "
            f"max_validation_samples={self.config['data'].get('max_validation_samples')} | "
            f"steps_per_epoch={self.config['training'].get('steps_per_epoch')} | "
            f"validation_steps={self.config['training'].get('validation_steps')}"
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['training']['num_epochs']):
            # Training
            train_losses = self.train_epoch(train_dataloader, epoch)
            
            # Validation
            val_losses = self.validate(val_dataloader, epoch)
            
            # Update schedulers
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            # Logging
            print(f'Epoch {epoch}:')
            print(f'  Train Loss: {train_losses["total"]:.4f}')
            print(f'  Val Loss: {val_losses["total"]:.4f}')
            
            if self.config.get('wandb', {}).get('enabled', False):
                wandb.log({
                    'epoch': epoch,
                    'train/total_loss': train_losses['total'],
                    'train/reconstruction_loss': train_losses['reconstruction'],
                    'train/timbre_loss': train_losses['timbre'],
                    'train/pitch_loss': train_losses['pitch'],
                    'train/asr_loss': train_losses['asr'],
                    'val/total_loss': val_losses['total'],
                    'val/reconstruction_loss': val_losses['reconstruction'],
                    'val/timbre_loss': val_losses['timbre'],
                    'val/pitch_loss': val_losses['pitch'],
                    'val/asr_loss': val_losses['asr']
                })
            
            # Save the checkpoint
            is_best = val_losses['total'] < best_val_loss
            if is_best:
                best_val_loss = val_losses['total']
            
            if epoch % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch, 0, is_best)
        
        print('Training completed!')


def main():
    parser = argparse.ArgumentParser(description='MulliVC training')
    parser.add_argument('--config', type=str, default='configs/mullivc_config.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to the checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to run')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size override')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='DataLoader worker count override')
    parser.add_argument('--max-train-samples', type=int, default=None,
                       help='Maximum number of training samples')
    parser.add_argument('--max-val-samples', type=int, default=None,
                       help='Maximum number of validation samples')
    parser.add_argument('--steps-per-epoch', type=int, default=None,
                       help='Maximum number of training batches per epoch')
    parser.add_argument('--validation-steps', type=int, default=None,
                       help='Maximum number of validation batches per epoch')
    parser.add_argument('--enable-cycle-consistency', action='store_true',
                       help='Enable the cycle consistency stage')
    parser.add_argument('--disable-wandb', action='store_true',
                       help='Disable W&B even if it is enabled in the config')
    
    args = parser.parse_args()

    overrides = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'max_train_samples': args.max_train_samples,
        'max_val_samples': args.max_val_samples,
        'steps_per_epoch': args.steps_per_epoch,
        'validation_steps': args.validation_steps,
        'enable_cycle_consistency': args.enable_cycle_consistency,
        'disable_wandb': args.disable_wandb,
    }
    
    # Create the trainer
    trainer = MulliVCTrainer(args.config, overrides=overrides)
    
    # Resume from a checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Checkpoint loaded from {args.resume}')
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
