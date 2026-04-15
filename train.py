"""
Training script for MulliVC.
"""
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import yaml
import os
import wandb
from tqdm import tqdm
import argparse
from datetime import datetime, timezone
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
        lr_g = self.config['training']['learning_rate']
        lr_d = self.config['training'].get('learning_rate_d', lr_g)
        self.optimizer_g = optim.Adam(
            [
                {'params': self.model.content_encoder.parameters()},
                {'params': self.model.timbre_encoder.parameters()},
                {'params': self.model.fine_grained_conformer.parameters()},
                {'params': self.model.mel_decoder.parameters()}
            ],
            lr=lr_g,
            betas=(self.config['training']['beta1'], self.config['training']['beta2']),
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=lr_d,
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

        # AMP (mixed precision) support
        self.use_amp = self.config['training'].get('amp', False)
        self.scaler_g = GradScaler(enabled=self.use_amp)
        self.scaler_d = GradScaler(enabled=self.use_amp)
        self.grad_clip_norm = self.config['training'].get('gradient_clip_norm', 0.0)
        
        # Audio processor
        self.audio_processor = AudioProcessor(self.config)

        self.log_dir = Path(self.config['paths']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / 'metrics.jsonl'
        self.progress_path = self.log_dir / 'progress.json'
        self.best_val_loss = float('inf')
        self.best_epoch = None
        self.progress_state: Dict[str, Any] = {}
        self._write_progress(
            status='initializing',
            device=str(self.device),
            config_path=config_path,
            steps_per_epoch=self.config['training'].get('steps_per_epoch'),
            validation_steps=self.config['training'].get('validation_steps'),
            num_epochs=self.config['training'].get('num_epochs'),
            batch_size=self.config['data'].get('batch_size'),
        )
        
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

    def _safe_len(self, dataloader: DataLoader) -> Optional[int]:
        """Returns len(dataloader) when available."""
        try:
            return len(dataloader)
        except TypeError:
            return None

    def _losses_to_scalars(self, losses: Dict[str, Any]) -> Dict[str, float]:
        """Converts tensors in a loss dictionary into Python floats."""
        scalars = {}
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                scalars[key] = float(value.detach().item())
            else:
                scalars[key] = float(value)
        return scalars

    def _append_metrics(self, record: Dict[str, Any]):
        """Appends one structured metrics record to disk."""
        payload = dict(record)
        payload['timestamp'] = datetime.now(timezone.utc).isoformat()
        with self.metrics_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, sort_keys=True) + '\n')

    def _write_progress(self, **updates: Any):
        """Writes the latest training progress snapshot to disk."""
        self.progress_state.update(updates)
        self.progress_state['updated_at'] = datetime.now(timezone.utc).isoformat()
        self.progress_state['best_val_loss'] = (
            None if self.best_val_loss == float('inf') else float(self.best_val_loss)
        )
        self.progress_state['best_epoch'] = self.best_epoch
        with self.progress_path.open('w', encoding='utf-8') as handle:
            json.dump(self.progress_state, handle, indent=2, sort_keys=True)

    def _bounded_num_batches(self, dataloader: DataLoader, limit: Optional[int]) -> int:
        """Returns the effective number of batches for this pass."""
        try:
            num_batches = len(dataloader)
        except TypeError:
            if limit is None:
                raise RuntimeError('Iterable dataloader requires an explicit batch limit.')
            return int(limit)

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

        self._write_progress(
            status='training',
            phase='train',
            current_epoch=epoch,
            current_batch=0,
            epoch_batches=num_batches,
        )

        with tqdm(total=num_batches, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                # Move data to the device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Train the generator -- stage-by-stage backward to reduce
                # peak GPU memory (all 3 stages kept the full graph alive).
                self.optimizer_g.zero_grad()
                stage_funcs = [
                    self.model._training_step_1,
                    self.model._training_step_2,
                ]
                if self.model.enable_cycle_consistency:
                    stage_funcs.append(self.model._training_step_3)
                num_stages = len(stage_funcs)

                accumulated_losses_g: Dict[str, float] = {}
                for stage_fn in stage_funcs:
                    with autocast('cuda', enabled=self.use_amp):
                        stage_losses = stage_fn(batch)
                    self.scaler_g.scale(stage_losses['total'] / num_stages).backward()
                    for k, v in stage_losses.items():
                        accumulated_losses_g[k] = accumulated_losses_g.get(k, 0.0) + v.item() / num_stages
                    del stage_losses
                    torch.cuda.empty_cache()

                if self.grad_clip_norm > 0:
                    self.scaler_g.unscale_(self.optimizer_g)
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in self.optimizer_g.param_groups for p in group['params']],
                        self.grad_clip_norm,
                    )
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()
                
                # Build a "losses_g" dict with plain floats for logging
                losses_g = accumulated_losses_g
                
                # Train the discriminator
                self.optimizer_d.zero_grad()
                with autocast('cuda', enabled=self.use_amp):
                    losses_d = self._train_discriminator(batch)
                self.scaler_d.scale(losses_d['total']).backward()
                if self.grad_clip_norm > 0:
                    self.scaler_d.unscale_(self.optimizer_d)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.discriminator.parameters(),
                        self.grad_clip_norm,
                    )
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
                
                # Free discriminator graph
                torch.cuda.empty_cache()
                
                # Update total losses (losses_g values are already floats)
                for key in total_losses:
                    if key in losses_g:
                        total_losses[key] += losses_g[key]
                    if key in losses_d:
                        total_losses[key] += losses_d[key].item()
                
                # Update the progress bar
                pbar.set_postfix({
                    'G_Loss': f'{losses_g["total"]:.4f}',
                    'D_Loss': f'{losses_d["total"].item():.4f}'
                })
                pbar.update(1)
                
                # Logging
                if batch_idx % self.config['training']['log_interval'] == 0:
                    self._log_metrics(epoch, batch_idx, losses_g, losses_d)
                    self._write_progress(
                        status='training',
                        phase='train',
                        current_epoch=epoch,
                        current_batch=batch_idx + 1,
                        epoch_batches=num_batches,
                        latest_generator_losses=self._losses_to_scalars(losses_g),
                        latest_discriminator_losses=self._losses_to_scalars(losses_d),
                    )
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches
        
        return total_losses
    
    def _train_discriminator(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Trains the discriminator on both self-recon and cross-speaker fakes."""
        source_audio = batch['audio']

        # Generate fake mels from cross-speaker conversion (stage 2 scenario),
        # which is what the discriminator actually needs to reject.
        target_timbre_audio = torch.roll(source_audio, 1, dims=0)
        with torch.no_grad():
            outputs = self.model.forward(source_audio, target_timbre_audio)
            generated_mel = outputs['generated_mel']
        
        # Real samples (target speaker's mel is what the output should match)
        real_mel = self.audio_processor.audio_to_mel(target_timbre_audio)
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
        generator_metrics = self._losses_to_scalars(losses_g)
        discriminator_metrics = self._losses_to_scalars(losses_d)

        if self.config.get('wandb', {}).get('enabled', False):
            metrics = {
                'epoch': epoch,
                'batch': batch_idx,
                'generator/total_loss': generator_metrics['total'],
                'generator/reconstruction_loss': generator_metrics['reconstruction'],
                'generator/timbre_loss': generator_metrics['timbre'],
                'generator/pitch_loss': generator_metrics['pitch'],
                'generator/asr_loss': generator_metrics['asr'],
                'discriminator/total_loss': discriminator_metrics['total'],
                'discriminator/adversarial_loss': discriminator_metrics['adversarial']
            }
            wandb.log(metrics)

        self._append_metrics({
            'kind': 'train_step',
            'epoch': epoch,
            'batch': batch_idx,
            'generator': generator_metrics,
            'discriminator': discriminator_metrics,
        })
    
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

        self._write_progress(
            status='validating',
            phase='validation',
            current_epoch=epoch,
            current_batch=0,
            epoch_batches=num_batches,
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, total=num_batches, desc='Validation')):
                if batch_idx >= num_batches:
                    break

                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                with autocast('cuda', enabled=self.use_amp):
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

                if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                    self._write_progress(
                        status='validating',
                        phase='validation',
                        current_epoch=epoch,
                        current_batch=batch_idx + 1,
                        epoch_batches=num_batches,
                    )
        
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
        try:
            train_dataloader = create_dataloader(self.config, split='train')
            val_dataloader = create_dataloader(self.config, split='validation', shuffle=False)

            train_batches = self._safe_len(train_dataloader)
            val_batches = self._safe_len(val_dataloader)
            train_mode = getattr(train_dataloader.dataset, 'dataset_mode', train_dataloader.dataset.__class__.__name__)
            val_mode = getattr(val_dataloader.dataset, 'dataset_mode', val_dataloader.dataset.__class__.__name__)

            print(
                f"Device: {self.device} | batch_size={self.config['data']['batch_size']} | "
                f"max_train_samples={self.config['data'].get('max_train_samples')} | "
                f"max_validation_samples={self.config['data'].get('max_validation_samples')} | "
                f"steps_per_epoch={self.config['training'].get('steps_per_epoch')} | "
                f"validation_steps={self.config['training'].get('validation_steps')}"
            )
            print(
                f"Train dataset: mode={train_mode} batches={train_batches if train_batches is not None else 'streaming'} | "
                f"Validation dataset: mode={val_mode} batches={val_batches if val_batches is not None else 'streaming'}"
            )

            self._write_progress(
                status='ready',
                phase='setup-complete',
                current_epoch=None,
                current_batch=None,
                train_dataset_mode=train_mode,
                validation_dataset_mode=val_mode,
                train_batches=train_batches,
                validation_batches=val_batches,
            )

            for epoch in range(self.config['training']['num_epochs']):
                train_losses = self.train_epoch(train_dataloader, epoch)
                val_losses = self.validate(val_dataloader, epoch)

                self.scheduler_g.step()
                self.scheduler_d.step()

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

                is_best = val_losses['total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses['total']
                    self.best_epoch = epoch

                if epoch % self.config['training']['save_interval'] == 0:
                    self.save_checkpoint(epoch, 0, is_best)

                self._append_metrics({
                    'kind': 'epoch_summary',
                    'epoch': epoch,
                    'train': train_losses,
                    'validation': val_losses,
                    'is_best': is_best,
                    'best_val_loss': self.best_val_loss,
                    'learning_rate_g': float(self.optimizer_g.param_groups[0]['lr']),
                    'learning_rate_d': float(self.optimizer_d.param_groups[0]['lr']),
                })
                self._write_progress(
                    status='epoch-complete',
                    phase='epoch-complete',
                    current_epoch=epoch,
                    current_batch=None,
                    completed_epochs=epoch + 1,
                    last_train_losses=train_losses,
                    last_validation_losses=val_losses,
                    latest_checkpoint=str(Path(self.config['paths']['checkpoint_dir']) / f'checkpoint_epoch_{epoch}_step_0.pt'),
                )

            self._write_progress(
                status='completed',
                phase='done',
                current_epoch=self.config['training']['num_epochs'] - 1,
                current_batch=None,
                completed_epochs=self.config['training']['num_epochs'],
            )
            print('Training completed!')
        except Exception as exc:
            self._write_progress(
                status='failed',
                phase='error',
                error=str(exc),
            )
            raise


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
