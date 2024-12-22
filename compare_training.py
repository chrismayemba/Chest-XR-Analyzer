import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm

from models.swin_transformer import XRayAnalyzer
from models.lora import LoRAConfig, save_lora_weights
from data.dataset import create_data_loaders
from train import Trainer

class ModelComparison:
    def __init__(self, data_dir, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loaders = create_data_loaders(data_dir)
        
        # Initialize models
        self.full_model = XRayAnalyzer(pretrained=True).to(self.device)
        self.lora_model = XRayAnalyzer(
            pretrained=True,
            lora_config=self._create_lora_config()
        ).to(self.device)
    
    def _create_lora_config(self):
        """Create LoRA configuration from yaml config"""
        lora_cfg = self.config['lora']
        return LoRAConfig(
            rank=lora_cfg['default']['rank'],
            alpha=lora_cfg['default']['alpha'],
            dropout=lora_cfg['default']['dropout'],
            layer_config=lora_cfg['layers']
        )
    
    def train_and_compare(self):
        """Train both models and compare performance"""
        results = {}
        
        # Train full model
        print("\nTraining full fine-tuning model...")
        full_trainer = Trainer(
            model=self.full_model,
            train_loader=self.data_loaders['train'],
            val_loader=self.data_loaders['val'],
            device=self.device,
            config=self.config['training']
        )
        full_metrics = self._train_model(full_trainer)
        results['full'] = full_metrics
        
        # Train LoRA model
        print("\nTraining LoRA model...")
        lora_trainer = Trainer(
            model=self.lora_model,
            train_loader=self.data_loaders['train'],
            val_loader=self.data_loaders['val'],
            device=self.device,
            config=self.config['training']
        )
        lora_metrics = self._train_model(lora_trainer)
        results['lora'] = lora_metrics
        
        # Compare and visualize results
        self._compare_results(results)
        
        return results
    
    def _train_model(self, trainer):
        """Train model and collect metrics"""
        start_time = time.time()
        
        metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'training_time': 0
        }
        
        for epoch in range(self.config['training']['epochs']):
            train_loss = trainer.train_epoch(epoch)
            val_loss, val_acc = trainer.validate(epoch)
            
            metrics['train_loss'].append(train_loss)
            metrics['val_loss'].append(val_loss)
            metrics['val_accuracy'].append(val_acc)
        
        metrics['training_time'] = time.time() - start_time
        return metrics
    
    def _compare_results(self, results):
        """Compare and visualize training results"""
        # Create comparison plots
        self._plot_metrics(results)
        
        # Print comparison summary
        print("\nTraining Comparison Summary:")
        print("-" * 50)
        for model_type, metrics in results.items():
            print(f"\n{model_type.upper()} Model:")
            print(f"Final validation accuracy: {metrics['val_accuracy'][-1]:.2f}%")
            print(f"Training time: {metrics['training_time']:.2f} seconds")
            print(f"Parameter count: {self._count_parameters(model_type):,}")
    
    def _plot_metrics(self, results):
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        
        # Loss comparison
        ax = axes[0]
        for model_type, metrics in results.items():
            ax.plot(metrics['train_loss'], label=f'{model_type} - Train')
            ax.plot(metrics['val_loss'], label=f'{model_type} - Val')
        ax.set_title('Loss Comparison')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # Accuracy comparison
        ax = axes[1]
        for model_type, metrics in results.items():
            ax.plot(metrics['val_accuracy'], label=model_type)
        ax.set_title('Validation Accuracy Comparison')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('training_comparison.png')
        plt.close()
    
    def _count_parameters(self, model_type):
        """Count trainable parameters"""
        model = self.full_model if model_type == 'full' else self.lora_model
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def evaluate_models(self):
        """Evaluate both models on test set"""
        results = {}
        
        for model_type, model in [('full', self.full_model), ('lora', self.lora_model)]:
            model.eval()
            test_metrics = self._evaluate_model(model)
            results[model_type] = test_metrics
            
            print(f"\n{model_type.upper()} Model Test Results:")
            print(f"Accuracy: {test_metrics['accuracy']:.2f}%")
            print(f"F1 Score: {test_metrics['f1']:.4f}")
        
        # Plot confusion matrices
        self._plot_confusion_matrices(results)
        
        return results
    
    def _evaluate_model(self, model):
        """Evaluate a single model on test set"""
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.data_loaders['test'], desc='Evaluating'):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = model(images)
            preds = outputs['classification'].argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': 100 * np.mean(np.array(all_preds) == np.array(all_labels)),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        
        return metrics
    
    def _plot_confusion_matrices(self, results):
        """Plot confusion matrices for both models"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        for idx, (model_type, metrics) in enumerate(results.items()):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
            axes[idx].set_title(f'{model_type.upper()} Model Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare training approaches')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    
    args = parser.parse_args()
    
    comparison = ModelComparison(args.data_dir, args.config)
    results = comparison.train_and_compare()
    test_results = comparison.evaluate_models()

if __name__ == '__main__':
    main()
