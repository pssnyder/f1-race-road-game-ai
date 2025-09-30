#!/usr/bin/env python3
"""
🚨 Emergency Model Cleanup Script
================================

Quickly compress large models and prepare for git-friendly storage
"""

import os
import sys
import zipfile
import json
from datetime import datetime

def compress_model(model_path, output_dir="models/export"):
    """Compress a model file with metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model info
    file_size = os.path.getsize(model_path) / (1024*1024)
    model_name = os.path.basename(model_path).replace('.pth', '')
    
    # Create compressed package
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    zip_name = f"{model_name}_compressed_{timestamp}.zip"
    zip_path = os.path.join(output_dir, zip_name)
    
    print(f"📦 Compressing {model_path} ({file_size:.1f}MB)...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # Add the model file
        zipf.write(model_path, os.path.basename(model_path))
        
        # Add metadata
        metadata = {
            "original_path": model_path,
            "original_size_mb": file_size,
            "compression_date": datetime.now().isoformat(),
            "notes": "Emergency compression for git storage",
            "usage": "Extract and load with torch.load()"
        }
        
        zipf.writestr('metadata.json', json.dumps(metadata, indent=2))
        
        # Add README
        readme = f"""
# F1 Racing AI Model Package
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- Original Size: {file_size:.1f} MB
- Original Path: {model_path}
- Compression Date: {datetime.now().strftime('%Y-%m-%d')}

## Usage
1. Extract this zip file
2. Load the model with PyTorch:
   ```python
   import torch
   model = torch.load('{os.path.basename(model_path)}', map_location='cpu')
   ```

## Notes
This model was compressed for git-friendly storage.
Original large file should be kept in .gitignore.
"""
        zipf.writestr('README.md', readme)
    
    # Check compression results
    compressed_size = os.path.getsize(zip_path) / (1024*1024)
    compression_ratio = (1 - compressed_size/file_size) * 100
    
    print(f"✅ Compressed successfully!")
    print(f"   📊 Size: {file_size:.1f}MB → {compressed_size:.1f}MB")
    print(f"   📉 Reduction: {compression_ratio:.1f}%")
    print(f"   📂 Output: {zip_path}")
    
    return zip_path, compressed_size

def main():
    # Find large models
    large_models = []
    
    for root, dirs, files in os.walk("models"):
        for file in files:
            if file.endswith('.pth'):
                path = os.path.join(root, file)
                size_mb = os.path.getsize(path) / (1024*1024)
                if size_mb > 50:  # Models larger than 50MB
                    large_models.append((path, size_mb))
    
    if not large_models:
        print("✅ No large models found!")
        return
    
    print(f"🔍 Found {len(large_models)} large models:")
    for path, size in large_models:
        print(f"   📄 {path} ({size:.1f}MB)")
    
    print("\n🚨 These models are too large for GitHub!")
    print("💡 Compressing them for git-friendly storage...")
    
    total_original = 0
    total_compressed = 0
    
    for model_path, original_size in large_models:
        zip_path, compressed_size = compress_model(model_path)
        total_original += original_size
        total_compressed += compressed_size
        print()
    
    # Summary
    total_savings = total_original - total_compressed
    savings_percent = (total_savings / total_original) * 100
    
    print("📊 COMPRESSION SUMMARY:")
    print(f"   📦 Total Original: {total_original:.1f}MB")
    print(f"   📦 Total Compressed: {total_compressed:.1f}MB") 
    print(f"   💾 Space Saved: {total_savings:.1f}MB ({savings_percent:.1f}%)")
    
    # Recommendations
    print("\n💡 NEXT STEPS:")
    print("1. Add compressed models to git:")
    print("   git add models/export/*.zip")
    print()
    print("2. The original .pth files are already in .gitignore")
    print("3. Consider moving large .pth files to external storage")
    print("4. Use 'python trainer.py' → models for ongoing management")

if __name__ == "__main__":
    main()