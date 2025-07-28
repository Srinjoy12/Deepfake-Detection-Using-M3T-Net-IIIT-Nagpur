# Deepfake Detector Deployment Guide

## 🚀 Deployment Architecture

- **Frontend**: Next.js app deployed on Vercel
- **Backend**: FastAPI app deployed on Railway (with GPU support)

## 📋 Prerequisites

1. GitHub account
2. Vercel account (free)
3. Railway account (free tier available)

## 🔧 Backend Deployment (Railway)

### Step 1: Deploy to Railway

1. Go to [Railway](https://railway.app)
2. Sign up/Login with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will auto-detect the Dockerfile in `/backend`

### Step 2: Configure Environment Variables

In Railway dashboard:
- Set `PORT` = `8000` (usually auto-detected)
- Add any other environment variables if needed

### Step 3: Configure Domain

1. In Railway dashboard, go to your service
2. Click "Settings" → "Domains"
3. Generate a domain or add custom domain
4. Note the URL (e.g., `https://your-app.railway.app`)

## 🌐 Frontend Deployment (Vercel)

### Step 1: Deploy to Vercel

1. Go to [Vercel](https://vercel.com)
2. Sign up/Login with GitHub
3. Click "New Project"
4. Import your GitHub repository
5. Vercel will auto-detect Next.js

### Step 2: Configure Environment Variables

In Vercel dashboard:
1. Go to Project Settings → Environment Variables
2. Add: `NEXT_PUBLIC_API_URL` = `https://your-backend-url.railway.app`

### Step 3: Deploy

1. Click "Deploy"
2. Vercel will build and deploy automatically
3. Get your frontend URL (e.g., `https://your-app.vercel.app`)

## 🔗 Update CORS

After deployment, update backend CORS in `main.py`:

```python
allow_origins=[
    "http://localhost:3000", 
    "http://localhost:3002", 
    "https://your-app.vercel.app"  # Add your Vercel URL
]
```

## 🧪 Testing

1. Visit your Vercel URL
2. Upload a video file
3. Check that analysis works end-to-end

## 📊 Monitoring

- **Railway**: Monitor backend logs and performance
- **Vercel**: Monitor frontend performance and build logs

## 💡 Tips

1. **Model Files**: Railway handles large files (your .pth models)
2. **GPU Support**: Railway Pro plan offers GPU instances for faster inference
3. **Scaling**: Both platforms auto-scale based on traffic
4. **Custom Domains**: Both support custom domains

## 🚨 Troubleshooting

### CORS Issues
- Ensure frontend URL is in backend CORS origins
- Check environment variables are set correctly

### Model Loading Issues
- Verify model files are included in deployment
- Check Railway build logs for errors

### Performance Issues
- Consider Railway Pro for GPU acceleration
- Optimize model loading and inference

## 💰 Cost Estimates

- **Vercel**: Free tier (hobby projects)
- **Railway**: $5/month for starter plan, more for GPU instances

---

**Need help?** Check the logs in both Railway and Vercel dashboards for detailed error information. 