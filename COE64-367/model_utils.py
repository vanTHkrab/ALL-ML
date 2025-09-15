"""
Enhanced Model Utilities for Mango Volume Estimation
Includes evaluation, inference, and visualization tools
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
import os
import glob
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Import from main models file
from models_torch import (
    EnhancedMangoModel, MangoDataset, get_transforms, 
    load_table, metrics, MODEL_NAME, IMG_SIZE, DEVICE
)

class ModelEvaluator:
    """Enhanced model evaluation and inference utilities"""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device or DEVICE
        self.model = None
        self.config = None
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.config = checkpoint.get('config', {})
        model_name = self.config.get('model_name', 'efficientnet_b3')
        
        # Initialize model
        self.model = EnhancedMangoModel(
            model_name=model_name,
            freeze_backbone=False,  # Unfreeze for inference
            dropout_rate=0.0  # No dropout during inference
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        print(f"✅ Model loaded successfully! Config: {self.config}")
        
    def predict_single(self, image_path: str, weight_g: float) -> Dict:
        """Predict volume for a single image"""
        # Load and preprocess image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return {"error": str(e)}
            
        transform = get_transforms(is_training=False)
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Prepare weight (normalize same as training)
        weight_tensor = torch.tensor([[weight_g / 100.0]], dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            pred = self.model(img_tensor, weight_tensor)
            volume_pred = float(pred.squeeze())
            
        return {
            "predicted_volume": volume_pred,
            "input_weight": weight_g,
            "image_path": image_path
        }
    
    def evaluate_dataset(self, csv_path: str, img_dir: str) -> Dict:
        """Evaluate model on a dataset"""
        print(f"Evaluating on dataset: {csv_path}")
        
        # Load data
        df = load_table(csv_path)
        transform = get_transforms(is_training=False)
        dataset = MangoDataset(df, img_dir, transform, is_training=False)
        
        # Predict all samples
        predictions = []
        targets = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                img, w, y, sample_id = dataset[i]
                
                img = img.unsqueeze(0).to(self.device)
                w = w.unsqueeze(0).to(self.device)
                
                pred = self.model(img, w)
                
                predictions.append(float(pred.squeeze()))
                targets.append(float(y.squeeze()))
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        
        results = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "predictions": predictions,
            "targets": targets,
            "n_samples": len(targets)
        }
        
        print(f"📊 Evaluation Results:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return results
    
    def visualize_predictions(self, results: Dict, save_path: str = None):
        """Create visualization plots"""
        predictions = results['predictions']
        targets = results['targets']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Prediction vs Target scatter plot
        axes[0, 0].scatter(targets, predictions, alpha=0.6, s=50)
        axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Volume (ml)')
        axes[0, 0].set_ylabel('Predicted Volume (ml)')
        axes[0, 0].set_title(f'Predictions vs True Values (R² = {results["r2"]:.4f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = predictions - targets
        axes[0, 1].scatter(targets, residuals, alpha=0.6, s=50)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('True Volume (ml)')
        axes[0, 1].set_ylabel('Residuals (ml)')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals (ml)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Error Distribution (MAE = {results["mae"]:.2f})')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Absolute error vs true values
        abs_errors = np.abs(residuals)
        axes[1, 1].scatter(targets, abs_errors, alpha=0.6, s=50)
        axes[1, 1].set_xlabel('True Volume (ml)')
        axes[1, 1].set_ylabel('Absolute Error (ml)')
        axes[1, 1].set_title(f'Absolute Error vs True Values (RMSE = {results["rmse"]:.2f})')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Visualization saved to {save_path}")
        
        plt.show()
        
    def cross_validate(self, csv_path: str, img_dir: str, k_folds: int = 5) -> Dict:
        """Perform k-fold cross validation"""
        print(f"Performing {k_folds}-fold cross validation...")
        
        df = load_table(csv_path)
        n_samples = len(df)
        fold_size = n_samples // k_folds
        
        fold_results = []
        
        for fold in range(k_folds):
            print(f"\nFold {fold + 1}/{k_folds}")
            
            # Split data
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < k_folds - 1 else n_samples
            
            val_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            train_df = pd.concat([
                df.iloc[:start_idx],
                df.iloc[end_idx:]
            ]).reset_index(drop=True)
            
            # Evaluate on validation fold
            transform = get_transforms(is_training=False)
            val_dataset = MangoDataset(val_df, img_dir, transform, is_training=False)
            
            predictions = []
            targets = []
            
            with torch.no_grad():
                for i in range(len(val_dataset)):
                    img, w, y, _ = val_dataset[i]
                    img = img.unsqueeze(0).to(self.device)
                    w = w.unsqueeze(0).to(self.device)
                    
                    pred = self.model(img, w)
                    predictions.append(float(pred.squeeze()))
                    targets.append(float(y.squeeze()))
            
            # Calculate fold metrics
            predictions = np.array(predictions)
            targets = np.array(targets)
            
            fold_mae = mean_absolute_error(targets, predictions)
            fold_rmse = np.sqrt(mean_squared_error(targets, predictions))
            fold_r2 = r2_score(targets, predictions)
            
            fold_results.append({
                'fold': fold + 1,
                'mae': fold_mae,
                'rmse': fold_rmse,
                'r2': fold_r2,
                'n_samples': len(targets)
            })
            
            print(f"  MAE: {fold_mae:.2f}, RMSE: {fold_rmse:.2f}, R²: {fold_r2:.4f}")
        
        # Calculate overall statistics
        overall_results = {
            'fold_results': fold_results,
            'mean_mae': np.mean([r['mae'] for r in fold_results]),
            'std_mae': np.std([r['mae'] for r in fold_results]),
            'mean_rmse': np.mean([r['rmse'] for r in fold_results]),
            'std_rmse': np.std([r['rmse'] for r in fold_results]),
            'mean_r2': np.mean([r['r2'] for r in fold_results]),
            'std_r2': np.std([r['r2'] for r in fold_results])
        }
        
        print(f"\n📊 Cross-Validation Results:")
        print(f"  MAE: {overall_results['mean_mae']:.2f} ± {overall_results['std_mae']:.2f}")
        print(f"  RMSE: {overall_results['mean_rmse']:.2f} ± {overall_results['std_rmse']:.2f}")
        print(f"  R²: {overall_results['mean_r2']:.4f} ± {overall_results['std_r2']:.4f}")
        
        return overall_results

def compare_models(model_paths: List[str], csv_path: str, img_dir: str):
    """Compare multiple trained models"""
    results = {}
    
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('.pt', '')
        print(f"\n🔍 Evaluating {model_name}...")
        
        try:
            evaluator = ModelEvaluator(model_path)
            result = evaluator.evaluate_dataset(csv_path, img_dir)
            results[model_name] = result
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue
    
    # Create comparison table
    if results:
        comparison_df = pd.DataFrame({
            name: {
                'MAE': res['mae'],
                'RMSE': res['rmse'],
                'R²': res['r2'],
                'MAPE (%)': res['mape']
            }
            for name, res in results.items()
        }).T
        
        print(f"\n📊 Model Comparison:")
        print(comparison_df.round(4))
        
        # Find best model
        best_model = comparison_df['R²'].idxmax()
        print(f"\n🏆 Best model: {best_model} (R² = {comparison_df.loc[best_model, 'R²']:.4f})")
        
        return comparison_df
    
    return None

# Example usage functions
def demo_inference():
    """Demo function for single image inference"""
    # Example usage
    model_path = "best_efficientnet_b3_enhanced.pt"
    
    if os.path.exists(model_path):
        evaluator = ModelEvaluator(model_path)
        
        # Example prediction (adjust paths as needed)
        result = evaluator.predict_single(
            image_path="./data/prepared-train/i001_crop_group_crop.jpg",
            weight_g=45.5
        )
        print(f"Prediction result: {result}")
    else:
        print(f"Model file {model_path} not found. Train the model first!")

def demo_evaluation():
    """Demo function for dataset evaluation"""
    model_path = "best_efficientnet_b0_enhanced.pt"
    csv_path = "./data/mangosteen-train-3.csv"
    img_dir = "./data/prepared-train"
    
    if os.path.exists(model_path):
        evaluator = ModelEvaluator(model_path)
        
        # Evaluate on dataset
        results = evaluator.evaluate_dataset(csv_path, img_dir)
        
        # Create visualizations
        evaluator.visualize_predictions(results, "model_evaluation.png")
        
        # Perform cross-validation
        cv_results = evaluator.cross_validate(csv_path, img_dir, k_folds=5)
        
    else:
        print(f"Model file {model_path} not found. Train the model first!")

if __name__ == "__main__":
    print("🔧 Model Utilities Loaded!")
    print("Available functions:")
    print("  - demo_inference(): Test single image prediction")
    print("  - demo_evaluation(): Evaluate model on dataset")
    print("  - compare_models(): Compare multiple trained models")
