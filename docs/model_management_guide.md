# 🗂️ F1 Racing AI Model Management Guide

## 🎯 **Keeping Your Repository Clean & Budget-Friendly**

This guide shows you how to manage your AI models efficiently while keeping your GitHub repository clean and your LFS usage minimal.

## 📋 **Current GitHub Options for ML Models**

### 1. **Git LFS (Large File Storage)** 
- ✅ **Good for**: Small teams, occasional model storage
- ❌ **Limitations**: 1GB free, then $5/month per 50GB
- 💡 **Best Practice**: Use for final models only, not checkpoints

### 2. **GitHub Releases** 
- ✅ **Good for**: Final model distributions
- ✅ **Free**: Up to 2GB per release file
- 💡 **Best Practice**: Package final models as release assets

### 3. **Git Ignore + External Storage**
- ✅ **Good for**: Development workflow
- ✅ **Free**: No GitHub storage used
- 💡 **Best Practice**: Use for training checkpoints

### 4. **GitHub Packages** (Container Registry)
- ✅ **Good for**: Containerized model serving
- ❌ **Complexity**: Overkill for simple models
- 💡 **Best Practice**: Advanced deployment scenarios

## 🛠️ **Recommended Setup for F1 Racing AI**

### **Strategy: Hybrid Approach**

```bash
# 1. ALWAYS in .gitignore (training artifacts)
models/checkpoints/          # Training checkpoints
models/temp/                 # Temporary models
*.pth.temp                   # Temporary files
training_logs/               # Raw training logs
__pycache__/                # Python cache

# 2. TRACKED in Git (small, important files)
model_manager.py             # Management system
models/model_registry.json   # Model metadata
docs/model_guide.md         # Documentation

# 3. COMPRESSED for Git LFS (final models only)
models/final/*.zip          # Compressed final models
models/export/*.zip         # Exported model packages

# 4. GITHUB RELEASES (distribution)
# Use releases for major model versions
```

### **Enhanced .gitignore**

```gitignore
# =========================
# F1 Racing AI - Model Files
# =========================

# Training checkpoints (too large, frequently changing)
models/checkpoints/*.pth
models/temp/
*_checkpoint_episode_*.pth
*interrupted_episode*.pth

# Raw model files (use compressed versions instead)
*.pth
!models/final/*.zip  # Keep compressed finals

# Training artifacts
training_logs/
results/logs/*.log
*.tmp

# Python cache
__pycache__/
*.pyc
*.pyo

# Environment files
.env
.venv/
venv/

# IDE files
.vscode/settings.json
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db
desktop.ini

# =========================
# Keep These Files
# =========================
# model_manager.py           ✅ Keep - management system
# models/model_registry.json ✅ Keep - metadata only
# models/final/*.zip         ✅ Keep - compressed finals
# models/export/*.zip        ✅ Keep - distribution packages
# requirements.txt           ✅ Keep - dependencies
# README.md                  ✅ Keep - documentation
```

## 📦 **Model Storage Workflow**

### **1. Development Phase**
```bash
# Train model (creates checkpoint)
python trainer.py
# → models/checkpoints/ai_driver_checkpoint_episode_500.pth (ignored by git)

# Register model
python -c "from model_manager import ModelManager; m=ModelManager(); m.scan_and_register_untracked()"
# → Updates models/model_registry.json (tracked by git)
```

### **2. Milestone Phase**
```bash
# Export final model as package
python trainer.py
# Choose: models → export model package
# → models/export/f1_ai_model_abc123_20250929.zip (tracked by git)

# Cleanup old models
python trainer.py
# Choose: cleanup
# → Removes old checkpoints, keeps metadata
```

### **3. Release Phase**
```bash
# Create GitHub release
gh release create v1.0.0 \
  --title "F1 Racing AI v1.0 - Enhanced 7-Feature Model" \
  --notes "Major upgrade with faster movement and extended vision" \
  models/export/f1_ai_model_final_v1.zip

# Archive development models
python trainer.py
# Choose: models → archive old models
# → Creates compressed archives, removes originals
```

## 💰 **Budget-Friendly Storage Strategy**

### **Cost Breakdown**
- **GitHub Free**: Unlimited public repos, 1GB LFS
- **GitHub Pro** ($4/month): 2GB LFS included
- **LFS Packs**: $5/month per 50GB

### **Our Approach** (Keep costs near $0)
1. **Use model compression** - 50-90% size reduction
2. **Archive old models** - Only keep recent + best
3. **External storage for checkpoints** - OneDrive, Google Drive, etc.
4. **GitHub Releases for distribution** - Free 2GB per file

### **Size Management**
```python
# Typical F1 Racing AI model sizes:
Raw PyTorch model:     ~2-5 MB   # Full checkpoint data
Compressed model:      ~0.5-1 MB # Just weights
Export package:        ~1-2 MB   # Model + code + docs

# With 1GB LFS allowance:
Raw models: ~200-500 models     # If using LFS
Compressed: ~1000-2000 models   # Much better!
```

## 🔧 **Automated Model Management**

### **Setup Pre-commit Hook**
```bash
# .git/hooks/pre-commit
#!/bin/bash
echo "🔍 Checking model files..."

# Auto-compress large models
find models/ -name "*.pth" -size +1M -exec python -c "
from model_manager import ModelManager
import sys
manager = ModelManager()
manager.auto_compress('{}')
" \;

# Update registry
python -c "
from model_manager import ModelManager
manager = ModelManager()
manager.scan_and_register_untracked()
print('✅ Model registry updated')
"
```

### **Automated Cleanup Script**
```python
# scripts/model_maintenance.py
"""
Weekly model maintenance script
Run this to keep your model storage optimized
"""

from model_manager import ModelManager
import schedule
import time

def weekly_cleanup():
    manager = ModelManager()
    
    print("🧹 Weekly model maintenance...")
    
    # Register any new models
    registered = manager.scan_and_register_untracked()
    print(f"📝 Registered {len(registered)} new models")
    
    # Clean duplicates
    removed = manager.cleanup_duplicates()
    print(f"🗑️  Removed {len(removed)} duplicates")
    
    # Archive old models (keep last 5 and best 3)
    archived = manager.archive_old_models(keep_recent=5, keep_best=3)
    print(f"📦 Archived {len(archived)} old models")
    
    # Generate report
    print(manager.generate_report())

# Schedule weekly cleanup
schedule.every().sunday.at("02:00").do(weekly_cleanup)

if __name__ == "__main__":
    print("🤖 Model maintenance scheduler started")
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour
```

## 🚀 **Quick Start Guide**

### **1. Initial Setup**
```bash
# Add to .gitignore
echo "*.pth" >> .gitignore
echo "models/checkpoints/" >> .gitignore
echo "models/temp/" >> .gitignore

# Set up model manager
python trainer.py
# Choose: models → scan for models
```

### **2. Training Workflow**
```bash
# Train model
python trainer.py
# Choose: train → [your settings]

# After training completes:
python trainer.py  
# Choose: cleanup    # Quick cleanup
# Then: models → export model package    # Create distributable
```

### **3. Sharing Models**
```bash
# For teammates (small models)
git add models/export/model_package.zip
git commit -m "Add trained model package"

# For public release (any size)
gh release create v1.1.0 models/export/model_package.zip

# For external backup
python -c "
from model_manager import ModelManager
manager = ModelManager()
export_path = manager.export_model_package('models/final/best_model.pth')
print(f'Upload {export_path} to your cloud storage')
"
```

## 🎯 **Best Practices Summary**

### ✅ **DO**
- Use the model manager for all model operations
- Compress models before adding to git
- Keep detailed metadata in the registry
- Use GitHub Releases for final distributions
- Archive old models regularly
- Document your model versions

### ❌ **DON'T**
- Commit raw .pth files to git
- Store training checkpoints in git
- Upload models larger than 100MB to LFS without compression
- Keep duplicate models around
- Forget to update the model registry

### 💡 **Pro Tips**
- Set up automated cleanup to run weekly
- Use meaningful model names and tags
- Export model packages before major changes
- Keep a backup of your best models externally
- Monitor your LFS usage in GitHub settings

---

## 📞 **Need Help?**

Run the model management system:
```bash
python trainer.py
# Choose: models → full management report
```

This will give you a complete overview of your current model storage situation and recommendations for optimization!