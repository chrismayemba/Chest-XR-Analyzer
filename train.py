import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import time
from tqdm import tqdm

from models.swin_transformer import XRayAnalyzer, LoRAConfig, get_lora_params, save_lora_weights
from data.dataset import create_data_loaders
from dataset_loader import ROCODataset
from transformers import AutoProcessor

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=1e-3, checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.segmentation_loss = nn.BCELoss()
        self.anomaly_loss = nn.BCELoss()
        
        # Optimizer - only optimize LoRA parameters and task-specific heads
        self.optimizer = optim.AdamW(
            [
                {'params': get_lora_params(model), 'lr': learning_rate},
                {'params': model.classifier.parameters(), 'lr': learning_rate},
                {'params': model.segmentation_head.parameters(), 'lr': learning_rate},
                {'params': model.anomaly_head.parameters(), 'lr': learning_rate},
                {'params': model.interpretation_head.parameters(), 'lr': learning_rate}
            ],
            lr=learning_rate
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir='runs/xray_analyzer')
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(images)
                
                # Calculate losses
                cls_loss = self.classification_loss(
                    outputs['classification'], labels
                )
                seg_loss = self.segmentation_loss(
                    outputs['segmentation'], masks
                )
                anomaly_loss = self.anomaly_loss(
                    outputs['anomaly'], 
                    (masks.sum(dim=(1,2,3)) > 0).float().unsqueeze(1)
                )
                
                # Combined loss
                loss = cls_loss + seg_loss + 0.5 * anomaly_loss
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'cls_loss': f'{cls_loss.item():.4f}',
                    'seg_loss': f'{seg_loss.item():.4f}'
                })
                
                # Log to tensorboard
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/train', loss.item(), step)
                self.writer.add_scalar('Loss/classification', cls_loss.item(), step)
                self.writer.add_scalar('Loss/segmentation', seg_loss.item(), step)
                self.writer.add_scalar('Loss/anomaly', anomaly_loss.item(), step)
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            outputs = self.model(images)
            
            # Calculate losses
            cls_loss = self.classification_loss(
                outputs['classification'], labels
            )
            seg_loss = self.segmentation_loss(
                outputs['segmentation'], masks
            )
            anomaly_loss = self.anomaly_loss(
                outputs['anomaly'],
                (masks.sum(dim=(1,2,3)) > 0).float().unsqueeze(1)
            )
            
            loss = cls_loss + seg_loss + 0.5 * anomaly_loss
            val_loss += loss.item()
            
            # Calculate accuracy
            pred = outputs['classification'].argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
        
        val_loss /= len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Log validation metrics
        self.writer.add_scalar('Loss/validation', val_loss, epoch)
        self.writer.add_scalar('Accuracy/validation', accuracy, epoch)
        
        return val_loss, accuracy
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best performance
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f'\nEpoch {epoch}/{num_epochs}')
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, accuracy = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Val Accuracy: {accuracy:.2f}%')

def main():
    parser = argparse.ArgumentParser(description='Train X-ray Analyzer')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use for training (cuda/cpu)')
    parser.add_argument('--lora-rank', type=int, default=4,
                      help='Rank for LoRA adaptation')
    parser.add_argument('--lora-alpha', type=float, default=1.0,
                      help='Alpha scaling factor for LoRA')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                      help='Dropout probability for LoRA')
    args = parser.parse_args()
    
    # Create data loaders
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
    train_dataset = ROCODataset(data_dir=args.data_dir, processor=processor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Assuming a validation dataset is similarly set up
    val_dataset = ROCODataset(data_dir=args.data_dir, processor=processor)  # Adjust path if needed
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model with LoRA
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )
    model = XRayAnalyzer(pretrained=True, lora_config=lora_config)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device(args.device),
        learning_rate=args.lr
    )
    
    # Start training
    trainer.train(args.epochs)
    
    # Save LoRA weights separately
    save_lora_weights(model, 'checkpoints/lora_weights.pt')

if __name__ == '__main__':
    main()
