"""
🗂️ F1 Racing AI Model Management System
======================================

A comprehensive system for managing AI models efficiently while keeping 
your repository clean and your budget intact.

Features:
- Automatic model versioning and metadata tracking
- Smart model compression and archival
- Model comparison and performance analysis
- Git-friendly storage with size optimization
- Automatic cleanup of redundant models
- Export/Import for external storage solutions

Author: F1 Racing AI Project
"""

import os
import json
import time
import shutil
import hashlib
import zipfile
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

class ModelMetadata:
    """Track comprehensive metadata for each model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.creation_time = datetime.now().isoformat()
        self.file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        self.model_hash = self._calculate_hash()
        self.episodes_trained = 0
        self.best_score = 0
        self.average_score = 0
        self.feature_count = 0
        self.training_duration = 0
        self.exploration_final = 0.0
        self.training_config = {}
        self.performance_stats = {}
        self.tags = []
        self.notes = ""
        
    def _calculate_hash(self) -> str:
        """Calculate unique hash for model file"""
        hash_md5 = hashlib.md5()
        with open(self.model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()[:16]  # Short hash for readability
    
    def update_from_checkpoint(self, checkpoint_data: dict):
        """Update metadata from model checkpoint data"""
        if 'episode_scores' in checkpoint_data:
            scores = checkpoint_data['episode_scores']
            self.episodes_trained = len(scores)
            if scores:
                self.best_score = max(scores)
                self.average_score = sum(scores) / len(scores)
        
        if 'current_epsilon' in checkpoint_data:
            self.exploration_final = checkpoint_data['current_epsilon']
            
        # Try to determine feature count from model structure
        if 'main_network_state' in checkpoint_data:
            state_dict = checkpoint_data['main_network_state']
            for key in state_dict.keys():
                if 'net.0.weight' in key or '0.weight' in key:
                    self.feature_count = state_dict[key].shape[1]
                    break
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage"""
        return {
            'model_path': self.model_path,
            'creation_time': self.creation_time,
            'file_size_mb': self.file_size_mb,
            'model_hash': self.model_hash,
            'episodes_trained': self.episodes_trained,
            'best_score': self.best_score,
            'average_score': self.average_score,
            'feature_count': self.feature_count,
            'training_duration': self.training_duration,
            'exploration_final': self.exploration_final,
            'training_config': self.training_config,
            'performance_stats': self.performance_stats,
            'tags': self.tags,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ModelMetadata':
        """Create from dictionary"""
        metadata = cls.__new__(cls)  # Create without calling __init__
        for key, value in data.items():
            setattr(metadata, key, value)
        return metadata

class ModelManager:
    """Comprehensive model management system"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.metadata_file = os.path.join(models_dir, "model_registry.json")
        self.archive_dir = os.path.join(models_dir, "archive")
        self.export_dir = os.path.join(models_dir, "export")
        
        # Create directories
        os.makedirs(self.archive_dir, exist_ok=True)
        os.makedirs(self.export_dir, exist_ok=True)
        
        # Load existing metadata
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, ModelMetadata]:
        """Load model registry from disk"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    return {path: ModelMetadata.from_dict(meta) 
                           for path, meta in data.items()}
            except Exception as e:
                print(f"⚠️  Warning: Could not load registry: {e}")
        return {}
    
    def _save_registry(self):
        """Save model registry to disk"""
        try:
            data = {path: meta.to_dict() for path, meta in self.registry.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving registry: {e}")
    
    def register_model(self, model_path: str, tags: List[str] = None, 
                      notes: str = "", training_config: dict = None) -> ModelMetadata:
        """Register a new model with metadata"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create metadata
        metadata = ModelMetadata(model_path)
        metadata.tags = tags or []
        metadata.notes = notes
        metadata.training_config = training_config or {}
        
        # Try to extract information from the model file
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            metadata.update_from_checkpoint(checkpoint)
        except Exception as e:
            print(f"⚠️  Could not analyze model: {e}")
        
        # Register
        self.registry[model_path] = metadata
        self._save_registry()
        
        print(f"✅ Registered model: {os.path.basename(model_path)}")
        print(f"   📊 {metadata.episodes_trained} episodes, {metadata.feature_count} features")
        print(f"   🏆 Best score: {metadata.best_score}")
        
        return metadata
    
    def list_models(self, filter_tags: List[str] = None, 
                   sort_by: str = "creation_time") -> List[Tuple[str, ModelMetadata]]:
        """List all registered models with filtering and sorting"""
        models = list(self.registry.items())
        
        # Filter by tags
        if filter_tags:
            models = [(path, meta) for path, meta in models 
                     if any(tag in meta.tags for tag in filter_tags)]
        
        # Sort
        if sort_by == "creation_time":
            models.sort(key=lambda x: x[1].creation_time, reverse=True)
        elif sort_by == "performance":
            models.sort(key=lambda x: x[1].best_score, reverse=True)
        elif sort_by == "episodes":
            models.sort(key=lambda x: x[1].episodes_trained, reverse=True)
        elif sort_by == "size":
            models.sort(key=lambda x: x[1].file_size_mb, reverse=True)
        
        return models
    
    def get_model_info(self, model_path: str) -> Optional[ModelMetadata]:
        """Get detailed information about a specific model"""
        return self.registry.get(model_path)
    
    def compare_models(self, model_paths: List[str]) -> dict:
        """Compare multiple models side by side"""
        comparison = {
            'models': [],
            'summary': {}
        }
        
        for path in model_paths:
            if path in self.registry:
                meta = self.registry[path]
                comparison['models'].append({
                    'path': path,
                    'name': os.path.basename(path),
                    'episodes': meta.episodes_trained,
                    'best_score': meta.best_score,
                    'avg_score': meta.average_score,
                    'features': meta.feature_count,
                    'size_mb': meta.file_size_mb,
                    'creation_time': meta.creation_time
                })
        
        # Calculate summary statistics
        if comparison['models']:
            comparison['summary'] = {
                'total_models': len(comparison['models']),
                'best_performer': max(comparison['models'], key=lambda x: x['best_score'])['name'],
                'most_episodes': max(comparison['models'], key=lambda x: x['episodes'])['name'],
                'total_size_mb': sum(m['size_mb'] for m in comparison['models']),
                'feature_types': set(m['features'] for m in comparison['models'])
            }
        
        return comparison
    
    def archive_old_models(self, keep_recent: int = 5, keep_best: int = 3) -> List[str]:
        """Archive old models to save space"""
        archived = []
        models = self.list_models(sort_by="creation_time")
        
        # Identify models to keep
        keep_models = set()
        
        # Keep most recent
        for path, _ in models[:keep_recent]:
            keep_models.add(path)
        
        # Keep best performers
        best_models = self.list_models(sort_by="performance")
        for path, _ in best_models[:keep_best]:
            keep_models.add(path)
        
        # Archive the rest
        for path, meta in models:
            if path not in keep_models and os.path.exists(path):
                archive_name = f"{meta.model_hash}_{os.path.basename(path)}.zip"
                archive_path = os.path.join(self.archive_dir, archive_name)
                
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(path, os.path.basename(path))
                    # Include metadata
                    zipf.writestr('metadata.json', json.dumps(meta.to_dict(), indent=2))
                
                # Remove original file
                os.remove(path)
                archived.append(path)
                
                # Update registry to point to archive
                meta.model_path = archive_path
                meta.notes += f" [ARCHIVED on {datetime.now().strftime('%Y-%m-%d')}]"
        
        self._save_registry()
        return archived
    
    def export_model_package(self, model_path: str, 
                           include_code: bool = True) -> str:
        """Export a complete model package for external storage"""
        if model_path not in self.registry:
            raise ValueError(f"Model not registered: {model_path}")
        
        meta = self.registry[model_path]
        export_name = f"f1_ai_model_{meta.model_hash}_{datetime.now().strftime('%Y%m%d')}.zip"
        export_path = os.path.join(self.export_dir, export_name)
        
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add model file
            zipf.write(model_path, f"model/{os.path.basename(model_path)}")
            
            # Add metadata
            zipf.writestr('metadata.json', json.dumps(meta.to_dict(), indent=2))
            
            # Add README
            readme_content = f"""
# F1 Racing AI Model Package
Generated: {datetime.now().isoformat()}

## Model Information
- Episodes Trained: {meta.episodes_trained}
- Best Score: {meta.best_score}
- Feature Count: {meta.feature_count}
- File Size: {meta.file_size_mb:.1f} MB

## Usage
1. Extract this package
2. Load the model file with PyTorch
3. Use with F1 Racing environment

## Notes
{meta.notes}
"""
            zipf.writestr('README.md', readme_content)
            
            # Optionally include source code
            if include_code:
                for code_file in ['agent.py', 'environment.py', 'trainer.py']:
                    if os.path.exists(code_file):
                        zipf.write(code_file, f"source/{code_file}")
        
        print(f"📦 Exported model package: {export_name}")
        return export_path
    
    def cleanup_duplicates(self) -> List[str]:
        """Remove duplicate models based on hash"""
        removed = []
        hash_to_path = {}
        
        for path, meta in self.registry.items():
            if meta.model_hash in hash_to_path:
                # This is a duplicate
                existing_path = hash_to_path[meta.model_hash]
                existing_meta = self.registry[existing_path]
                
                # Keep the one with more training or better performance
                if (meta.episodes_trained > existing_meta.episodes_trained or
                    meta.best_score > existing_meta.best_score):
                    # Remove the existing one
                    if os.path.exists(existing_path):
                        os.remove(existing_path)
                    removed.append(existing_path)
                    del self.registry[existing_path]
                    hash_to_path[meta.model_hash] = path
                else:
                    # Remove this one
                    if os.path.exists(path):
                        os.remove(path)
                    removed.append(path)
                    del self.registry[path]
            else:
                hash_to_path[meta.model_hash] = path
        
        self._save_registry()
        return removed
    
    def scan_and_register_untracked(self) -> List[str]:
        """Scan models directory and register any untracked models"""
        registered = []
        
        for root, dirs, files in os.walk(self.models_dir):
            for file in files:
                if file.endswith('.pth'):
                    full_path = os.path.join(root, file)
                    if full_path not in self.registry:
                        # Auto-detect tags based on path
                        tags = []
                        if 'checkpoint' in file:
                            tags.append('checkpoint')
                        if 'final' in file:
                            tags.append('final')
                        if 'best' in file:
                            tags.append('best')
                        
                        try:
                            self.register_model(full_path, tags=tags, 
                                              notes="Auto-registered by scan")
                            registered.append(full_path)
                        except Exception as e:
                            print(f"⚠️  Could not register {file}: {e}")
        
        return registered
    
    def generate_report(self) -> str:
        """Generate a comprehensive model management report"""
        report = []
        report.append("🗂️ F1 Racing AI Model Management Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        total_models = len(self.registry)
        total_size = sum(meta.file_size_mb for meta in self.registry.values())
        total_episodes = sum(meta.episodes_trained for meta in self.registry.values())
        
        report.append(f"📊 Summary:")
        report.append(f"   Total Models: {total_models}")
        report.append(f"   Total Size: {total_size:.1f} MB")
        report.append(f"   Total Episodes Trained: {total_episodes:,}")
        report.append("")
        
        # Best performers
        best_models = self.list_models(sort_by="performance")[:3]
        report.append("🏆 Top Performers:")
        for i, (path, meta) in enumerate(best_models, 1):
            report.append(f"   {i}. {os.path.basename(path)}")
            report.append(f"      Score: {meta.best_score} | Episodes: {meta.episodes_trained}")
        report.append("")
        
        # Model types
        feature_counts = {}
        for meta in self.registry.values():
            feature_counts[meta.feature_count] = feature_counts.get(meta.feature_count, 0) + 1
        
        report.append("🔍 Model Types:")
        for features, count in sorted(feature_counts.items()):
            report.append(f"   {features}-feature models: {count}")
        report.append("")
        
        # Storage recommendations
        report.append("💡 Storage Recommendations:")
        if total_size > 100:
            report.append("   ⚠️  Consider archiving old models (>100MB total)")
        if total_models > 10:
            report.append("   🗂️  Consider cleanup of duplicate models")
        if len([m for m in self.registry.values() if m.episodes_trained < 100]) > 3:
            report.append("   🧹  Consider removing early experiment models")
        
        return "\n".join(report)

def main():
    """CLI interface for model management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="F1 Racing AI Model Manager")
    parser.add_argument('action', choices=['scan', 'list', 'report', 'archive', 'cleanup', 'export'],
                       help='Action to perform')
    parser.add_argument('--path', help='Model path for specific operations')
    parser.add_argument('--tags', help='Comma-separated tags for filtering')
    
    args = parser.parse_args()
    
    manager = ModelManager()
    
    if args.action == 'scan':
        print("🔍 Scanning for untracked models...")
        registered = manager.scan_and_register_untracked()
        print(f"✅ Registered {len(registered)} new models")
        
    elif args.action == 'list':
        filter_tags = args.tags.split(',') if args.tags else None
        models = manager.list_models(filter_tags=filter_tags)
        
        print("📋 Registered Models:")
        for path, meta in models:
            print(f"   📄 {os.path.basename(path)}")
            print(f"      Episodes: {meta.episodes_trained} | Score: {meta.best_score}")
            print(f"      Features: {meta.feature_count} | Size: {meta.file_size_mb:.1f}MB")
            if meta.tags:
                print(f"      Tags: {', '.join(meta.tags)}")
            print()
    
    elif args.action == 'report':
        print(manager.generate_report())
    
    elif args.action == 'archive':
        print("📦 Archiving old models...")
        archived = manager.archive_old_models()
        print(f"✅ Archived {len(archived)} models")
    
    elif args.action == 'cleanup':
        print("🧹 Cleaning up duplicates...")
        removed = manager.cleanup_duplicates()
        print(f"✅ Removed {len(removed)} duplicate models")
    
    elif args.action == 'export':
        if not args.path:
            print("❌ --path required for export")
            return
        
        export_path = manager.export_model_package(args.path)
        print(f"📦 Model exported to: {export_path}")

if __name__ == "__main__":
    main()