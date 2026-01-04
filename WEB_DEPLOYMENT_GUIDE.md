# Web Deployment Guide for Fake News Detection System

**Last Updated:** January 4, 2026

---

## 🌐 Quick Deploy Options

### Option 1: Streamlit Community Cloud (⭐ RECOMMENDED - FREE)

**Best for:** Free hosting, easy deployment, perfect for thesis projects

#### Steps:

1. **Create a GitHub Repository**
   ```bash
   # In your project folder
   git init
   git add .
   git commit -m "Initial commit - Fake News Detection System"
   ```

2. **Push to GitHub**
   - Create new repository on GitHub: https://github.com/new
   - Name it: `fake-news-detection`
   - Make it public (required for free hosting)

   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/fake-news-detection.git
   git branch -M main
   git push -u origin main
   ```

3. **Deploy to Streamlit Cloud**
   - Go to: https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `fake-news-detection`
   - Main file path: `app.py`
   - Click "Deploy"

   **Your app will be live at:**
   `https://YOUR_USERNAME-fake-news-detection.streamlit.app`

4. **Wait for Deployment** (5-10 minutes)
   - Streamlit will install all dependencies from `requirements.txt`
   - Models will be loaded automatically
   - You'll get a public URL!

**Pros:**
- ✅ Completely FREE
- ✅ Easy deployment (3 clicks)
- ✅ Automatic SSL (https)
- ✅ Perfect for thesis demos
- ✅ Auto-updates when you push to GitHub

**Cons:**
- ⚠️ Sleeps after inactivity (wakes in ~30 seconds)
- ⚠️ Limited resources (1GB RAM)
- ⚠️ May be slower than paid options

---

### Option 2: Hugging Face Spaces (FREE)

**Best for:** ML projects, free hosting, no sleep mode

#### Steps:

1. **Create Hugging Face Account**
   - Go to: https://huggingface.co/join
   - Sign up (free)

2. **Create a New Space**
   - Go to: https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `fake-news-detection`
   - SDK: Select **Streamlit**
   - License: MIT
   - Click "Create Space"

3. **Upload Your Files**
   - Upload all `.py` files
   - Upload `requirements.txt`
   - Upload `models/` folder (all model files)
   - Upload `data/` folder (CSV files)
   - Create `app.py` as main file

4. **Your App is Live!**
   `https://huggingface.co/spaces/YOUR_USERNAME/fake-news-detection`

**Pros:**
- ✅ FREE forever
- ✅ No sleep mode
- ✅ Good for ML projects
- ✅ Community support

**Cons:**
- ⚠️ Limited CPU (slower than paid)
- ⚠️ Basic analytics only

---

### Option 3: Render (FREE tier available)

**Best for:** More control, better performance

#### Steps:

1. **Push to GitHub** (same as Option 1, steps 1-2)

2. **Create Render Account**
   - Go to: https://render.com/
   - Sign up with GitHub

3. **Create New Web Service**
   - Click "New" → "Web Service"
   - Connect GitHub repository
   - Name: `fake-news-detection`
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

4. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes

   **Your app will be at:**
   `https://fake-news-detection.onrender.com`

**Pros:**
- ✅ FREE tier available
- ✅ Better performance than Streamlit Cloud
- ✅ Custom domains
- ✅ No sleep on paid tier

**Cons:**
- ⚠️ Free tier sleeps after 15 min inactivity
- ⚠️ Slower build times

---

### Option 4: Heroku (PAID - $5/month minimum)

**Best for:** Production apps, professional hosting

#### Steps:

1. **Install Heroku CLI**
   - Download: https://devcenter.heroku.com/articles/heroku-cli
   - Install

2. **Create Procfile**
   Already created! (see files below)

3. **Deploy**
   ```bash
   heroku login
   heroku create fake-news-detection-app
   git push heroku main
   ```

**Pros:**
- ✅ No sleep mode (on paid tier)
- ✅ Professional hosting
- ✅ Great performance
- ✅ Custom domains

**Cons:**
- ❌ Costs $5+/month
- ⚠️ Requires credit card

---

## 📝 Required Files for Deployment

### 1. Procfile (for Heroku)
```
web: sh setup.sh && streamlit run app.py
```

### 2. setup.sh (for Heroku)
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

### 3. .streamlit/config.toml
Already created! ✅

### 4. requirements.txt
Already exists! ✅

---

## 🚀 Running Locally (For Testing)

Since `streamlit` is not in PATH, use:

```bash
# Run with Python module
python -m streamlit run app.py
```

**Or add to PATH:**

1. Find Python Scripts folder:
   ```bash
   python -c "import sys; print(sys.prefix + '\\Scripts')"
   ```

2. Add to Windows PATH:
   - Copy the path from above
   - Search "Environment Variables" in Windows
   - Edit PATH
   - Add the Scripts path
   - Restart PowerShell

3. Then you can use:
   ```bash
   streamlit run app.py
   ```

---

## 📦 Deployment Checklist

Before deploying, ensure:

- [x] All models trained: `python train_models.py`
- [x] Requirements.txt updated
- [x] .gitignore created
- [x] Large files < 100MB (GitHub limit)
- [ ] Test locally: `python -m streamlit run app.py`
- [ ] Create GitHub repository
- [ ] Choose deployment platform
- [ ] Deploy!

---

## ⚠️ Important Notes

### Model Files Size

Your model files are large. Check sizes:

```bash
# Check model sizes
ls -lh models/*.pkl
```

If any file > 100MB:
- **GitHub won't accept it**
- **Solution:** Use Git LFS (Large File Storage)

```bash
# Install Git LFS
git lfs install
git lfs track "models/*.pkl"
git add .gitattributes
git commit -m "Track model files with LFS"
```

### Data Privacy

- ❌ **DO NOT** include sensitive data in public repos
- ✅ Training data (Fake.csv, True.csv) is public - OK to include
- ✅ Models are fine - they don't contain original data

---

## 🎯 Recommended Deployment Strategy

**For Thesis/Academic Use:**

1. **Start with Streamlit Community Cloud** (FREE)
   - Easiest deployment
   - Perfect for demos
   - Good enough for thesis presentations

2. **Later, upgrade to Hugging Face Spaces** (FREE)
   - No sleep mode
   - Better for long-term hosting
   - ML-focused community

3. **For production, use Render or Heroku** (PAID)
   - Only if you need reliability
   - Better performance
   - Professional use

---

## 🌟 Example Deployment URLs

After deployment, your app will be accessible at:

- **Streamlit Cloud:** `https://YOUR_USERNAME-fake-news-detection.streamlit.app`
- **Hugging Face:** `https://huggingface.co/spaces/YOUR_USERNAME/fake-news-detection`
- **Render:** `https://fake-news-detection.onrender.com`
- **Heroku:** `https://fake-news-detection-app.herokuapp.com`

---

## 📊 Performance Expectations

| Platform | Cold Start | Response Time | Uptime |
|----------|------------|---------------|--------|
| Streamlit Cloud | 30s | 1-2s | Good (sleeps) |
| Hugging Face | None | 2-3s | Excellent |
| Render (free) | 30s | 1-2s | Good (sleeps) |
| Render (paid) | None | <1s | Excellent |
| Heroku | None | <1s | Excellent |

---

## 🐛 Troubleshooting Deployment

### Issue: Build fails on Streamlit Cloud

**Solution:** Check logs and ensure:
- `requirements.txt` is complete
- Model files are accessible
- No syntax errors in code

### Issue: App crashes with "Out of Memory"

**Solution:**
- Models too large for free tier
- Solution: Use Hugging Face Spaces (more RAM)
- Or optimize models (reduce features)

### Issue: LIME is too slow on deployed app

**Solution:** Reduce LIME samples in `components/visualizations.py`:
```python
num_samples=500  # Instead of 1000
```

---

## 📱 Mobile Responsiveness

Streamlit apps work on mobile! Your 3-tab dashboard will automatically adapt.

---

## 🔒 Security Considerations

**For Public Deployment:**

1. ✅ No API keys needed - good!
2. ✅ No database - good!
3. ✅ Models are safe to share
4. ⚠️ Consider rate limiting for production
5. ⚠️ Monitor usage to prevent abuse

---

## 📈 Analytics & Monitoring

**Streamlit Cloud:**
- Built-in analytics dashboard
- See visitor counts, usage stats

**Hugging Face:**
- View space analytics
- Track model downloads

**Custom Analytics:**
- Add Google Analytics (optional)
- Track predictions in session state

---

## 🎓 For Your Thesis

**Include in thesis:**

1. **Deployment Section:**
   - "The system is deployed at: [URL]"
   - "Accessible via web browser"
   - "No installation required for users"

2. **Screenshots:**
   - Live web app URL
   - Mobile responsive view
   - Public accessibility

3. **Discussion:**
   - "Production-ready deployment"
   - "Accessible to researchers globally"
   - "Scalable architecture"

---

## 🚀 Quick Start Deployment (5 Minutes)

**Fastest way to get your app online:**

```bash
# 1. Initialize git (if not already)
git init
git add .
git commit -m "Deploy fake news detection"

# 2. Create GitHub repo and push
# Go to github.com/new
# Copy the commands GitHub gives you

# 3. Go to share.streamlit.io
# Sign in with GitHub
# Deploy in 3 clicks!
```

**Done! Your app is live! 🎉**

---

## 📞 Need Help?

- Streamlit Docs: https://docs.streamlit.io/
- Deployment Guide: https://docs.streamlit.io/deploy/streamlit-community-cloud
- Community Forum: https://discuss.streamlit.io/

---

**Good luck with deployment! 🚀**

Your thesis project will be accessible worldwide! 🌍

---

*Generated: January 4, 2026*
*Status: Ready for Web Deployment*
