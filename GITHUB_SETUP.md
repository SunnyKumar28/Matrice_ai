# GitHub Setup Instructions

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the repository details:
   - **Repository name**: `Ultra-Optimized-Real-Time-Vision-Streaming-System` (or your preferred name)
   - **Description**: "High-performance real-time video inference system using YOLOv8 for object detection"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

## Step 2: Push Your Code to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
cd /home/sunny-gupta/Matrice

# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git

# Push your code
git push -u origin main
```

### Alternative: Using SSH (if you have SSH keys set up)

```bash
git remote add origin git@github.com:YOUR_USERNAME/REPOSITORY_NAME.git
git push -u origin main
```

## Step 3: Verify

1. Go to your GitHub repository page
2. You should see all your files:
   - `server.py`
   - `client.py`
   - `README.md`
   - `requirements.txt`
   - `COMMANDS.md`
   - `.gitignore`

## Important Notes

### Files NOT Uploaded (by design):
- `venv/` - Virtual environment (users will create their own)
- `results/` - Output files (excluded from git)
- `*.json` - Result files (excluded)
- `*.pt` - Model files (large, downloaded automatically)
- `yolov8n.pt` - YOLOv8 model (downloaded on first run)

### Model Files:
The YOLOv8 model (`yolov8n.pt`) is **not** included in the repository because:
- It's large (~6.5MB)
- It's automatically downloaded by ultralytics on first run
- Users can specify different models (yolov8n, yolov8s, yolov8m, etc.)

## Adding a License (Optional)

If you want to add a license:

```bash
# Example: MIT License
# Create LICENSE file or use GitHub's web interface
```

## Repository Badges (Optional)

You can add badges to your README.md:

```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)
```

## Troubleshooting

### If you get authentication errors:
```bash
# Use GitHub CLI or Personal Access Token
# Or set up SSH keys: https://docs.github.com/en/authentication
```

### If you need to update the remote URL:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git
```

### To check your remote:
```bash
git remote -v
```

## Next Steps After Uploading

1. **Add topics/tags** to your repository on GitHub (e.g., `yolov8`, `computer-vision`, `real-time`, `object-detection`)
2. **Update README** if needed with more details
3. **Create releases** for version tags if you want
4. **Add collaborators** if working with a team

