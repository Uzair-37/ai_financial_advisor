# 🚀 Deployment Guide

## Quick Deploy to Streamlit Cloud

### 1. Upload to GitHub
- Create new repository on GitHub
- Upload all files from this folder
- Make repository public

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file: `streamlit_app.py`
6. Click "Deploy!"

### 3. Your app is live!
- URL: `https://your-username-repo-name.streamlit.app`
- Share this link with anyone

## 🔑 Enable AI Features

### For Users:
1. Get free API key from [together.ai](https://together.ai)
2. Enter in app sidebar under "Setup AI"
3. Enjoy Llama 4 explanations!

### For Developers (Streamlit Cloud):
1. Go to your app settings
2. Add secret: `TOGETHER_API_KEY = "your_key"`
3. Restart app

## 🔧 Alternative Deployments

### Heroku
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### Local Testing
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 📊 App Features

- ✅ Works immediately without API key
- ✅ Real-time stock data
- ✅ Interactive charts
- ✅ Portfolio analysis
- ✅ Educational content
- 🦙 AI explanations (with API key)

Your AI Financial Advisor is ready for the world! 🌟