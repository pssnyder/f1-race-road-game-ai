# 🌐 F1 Race AI Web Dashboard

Real-time web-based monitoring for your F1 Race AI training progress! 

## 🚀 Quick Start

### Option 1: From Main Menu (Recommended)
```bash
python train_ai.py
# Choose 'dashboard' from the menu
```

### Option 2: Direct Launch
```bash
python dashboard_simple.py
```

### Option 3: Advanced Flask Dashboard (requires Flask)
```bash
pip install flask
python web_dashboard/app.py
```

Then visit: **http://localhost:5000**

## ✨ Features

### 📊 **Real-Time Monitoring**
- 🏎️ **Training Status**: See if AI is currently training or idle
- 📈 **Live Charts**: Auto-updating training progress charts
- ⚡ **Statistics**: Episodes completed, best scores, averages
- 🧠 **Learning Metrics**: Exploration rate, training loss

### 🎯 **Smart Updates**
- 🔄 **Auto-refresh**: Updates every 5 seconds automatically
- 📊 **Chart Sync**: Shows latest training charts as they're generated
- 💾 **Model Tracking**: Lists recent models and checkpoints
- ⏱️ **Timestamp**: Shows when data was last updated

### 🖥️ **User-Friendly Interface**
- 📱 **Responsive**: Works on desktop, tablet, and mobile
- 🎨 **Clean Design**: Modern, easy-to-read interface
- ⚡ **Fast Loading**: Lightweight and efficient
- 🌐 **No Dependencies**: Standalone version uses only Python built-ins

## 🎮 Usage Workflow

### Step 1: Start Dashboard
```bash
python train_ai.py
# Choose 'dashboard'
```

Browser opens automatically at: http://localhost:5000

### Step 2: Start Training (in another terminal)
```bash
python train_ai.py
# Choose 'train'
# Configure your training settings
```

### Step 3: Watch Progress
- 📊 Charts update automatically as training progresses
- 📈 Statistics refresh every few seconds
- 🎯 No need to switch between terminal and browser
- ⚡ Monitor multiple training runs

## 📊 Dashboard Sections

### 1. Status Bar
```
🟢 Training Active    | 2h 15m 32s     | 2024-09-28 14:30:25
Training Status       | Uptime         | Last Update
```

### 2. Training Statistics
- **Total Episodes**: Number of games played
- **Best Score**: Highest score achieved
- **Average Score**: Overall performance
- **Recent Average**: Last 100 episodes average
- **Exploration Rate**: Current randomness level (epsilon)
- **Training Loss**: Neural network learning progress

### 3. Recent Models
- **Final Models**: Completed training models
- **Checkpoints**: Intermediate saves during training
- **File Info**: Size, timestamp, type
- **Quick Access**: Easy identification of latest models

### 4. Training Progress Chart
- **Auto-updating**: Refreshes when new charts generated
- **Full Resolution**: Same charts as generated during training
- **Cache-busted**: Always shows latest version
- **Responsive**: Scales to fit screen size

## 🔧 Technical Details

### Standalone Version (`dashboard_simple.py`)
- **No Dependencies**: Uses Python's built-in HTTP server
- **Lightweight**: Minimal resource usage
- **Portable**: Works anywhere Python works
- **Fast**: Quick startup and response times

### Advanced Version (`web_dashboard/app.py`)
- **Flask-based**: More features and extensibility
- **Enhanced API**: Richer data endpoints
- **Template System**: Customizable HTML templates
- **Production Ready**: Scalable for multiple users

### Data Sources
- **Charts**: `results/charts/ai_training_progress.png`
- **Models**: `models/final/*.pth` and `models/checkpoints/*.pth`
- **Statistics**: Extracted from saved model files
- **Status**: Based on file modification times

## 🎯 Perfect For

### 👨‍💼 **Data Scientists**
- Monitor training progress without terminal clutter
- Quick visual assessment of model convergence
- Easy comparison of training runs
- Professional presentation of results

### 👩‍🏫 **Educators** 
- Demonstrate AI learning in real-time to students
- Visual explanation of training concepts
- Clean interface for classroom presentations
- Non-technical friendly display

### 👨‍💻 **Developers**
- Debug training issues quickly
- Monitor long-running training sessions
- Check progress remotely via web browser
- Integrate with existing workflows

### 🧒 **Students & Enthusiasts**
- Exciting visual feedback during training
- Easy to understand progress indicators
- Immediate gratification as AI improves
- Share progress with others via screenshots

## 🛠️ Customization

### Change Update Frequency
Edit the JavaScript in the dashboard HTML:
```javascript
setInterval(updateDashboard, 3000); // 3 seconds instead of 5
```

### Modify Port
```bash
# Edit dashboard_simple.py line ~290
port = 8080  # Change from 5000 to 8080
```

### Add Custom Metrics
Extend the `get_training_status()` function to include additional statistics from your model files.

## 🚨 Troubleshooting

### Dashboard Won't Start
```bash
# Check if port is in use
netstat -an | grep 5000

# Kill existing processes
pkill -f dashboard_simple.py

# Try different port
# Edit dashboard_simple.py and change port number
```

### Charts Not Updating
- Ensure training is generating charts (check `chart_update_frequency`)
- Verify `results/charts/` directory has write permissions
- Check training is actually running and progressing

### Statistics Not Showing
- Confirm model files exist in `models/` directory
- Verify PyTorch is installed (for model loading)
- Check model files aren't corrupted

### Browser Issues
- Try refreshing the page (Ctrl+F5)
- Clear browser cache
- Try different browser
- Check if JavaScript is enabled

## 🔗 Integration

### With Training
The dashboard monitors files that training naturally creates:
- No modification to training code needed
- Works with existing and new models
- Compatible with all training modes

### With Other Tools
- **Screenshots**: Easy progress sharing
- **Screen Recording**: Create training timelapses
- **Remote Access**: Monitor training from anywhere
- **Multiple Monitors**: Keep dashboard on second screen

## 🎉 Tips & Tricks

### Pro Tips
1. **Dual Terminal Setup**: One for training, one for dashboard
2. **Bookmark URL**: Quick access to http://localhost:5000
3. **Mobile Monitoring**: Check progress on your phone
4. **Screenshot Progress**: Share impressive training curves

### Performance Optimization
1. **Headless Training**: Use `show_training=False` for faster training
2. **Chart Frequency**: Balance between updates and performance
3. **Model Cleanup**: Remove old checkpoints to reduce load
4. **Browser Tabs**: Close unused tabs to save resources

---

## 🎯 Ready to Monitor Your AI Training?

```bash
python train_ai.py
# Choose 'dashboard'
# 🌐 Browser opens at http://localhost:5000
# 🏎️ Watch your AI learn to race in real-time!
```

The web dashboard brings your AI training to life with beautiful, real-time visualizations that make monitoring progress both easy and exciting! 🚀📊