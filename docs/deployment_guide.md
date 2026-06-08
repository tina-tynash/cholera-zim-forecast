# Deployment Guide

## Local Development

```bash
git clone https://github.com/YOUR_USERNAME/cholera-zim-forecast.git
cd cholera-zim-forecast
pip install -r requirements.txt
python data/synthetic/generate_synthetic.py --output-dir data/processed
python src/models/train_ensemble.py
streamlit run src/app/streamlit_app.py
```

## Streamlit Community Cloud (Free)

1. Push repository to GitHub (public).
2. Go to https://share.streamlit.io → New App.
3. Select repo, branch `main`, file `src/app/streamlit_app.py`.
4. Click Deploy. App live in ~2 minutes.

## Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```
Dashboard: http://localhost:8501  
API docs:   http://localhost:8000/docs

## AWS Free Tier

### S3 (data storage)
```bash
aws s3 mb s3://cholera-zim-data
aws s3 sync data/processed/ s3://cholera-zim-data/processed/
```

### EC2 (dashboard hosting)
```bash
# t2.micro — free tier eligible
aws ec2 run-instances --image-id ami-0abcdef1234567890 \
  --instance-type t2.micro --key-name your-key
# SSH in, clone repo, docker-compose up
```

### Lambda (API)
```bash
pip install mangum
# Add to api.py: handler = Mangum(app)
# Deploy with AWS SAM or Serverless Framework
```
