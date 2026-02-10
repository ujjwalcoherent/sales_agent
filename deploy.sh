#!/bin/bash
# Deploy to both GitHub and HF Spaces in one command
# Usage: bash deploy.sh "commit message"

MSG="${1:-Deploy update}"

# Push to GitHub
git add -A
git commit -m "$MSG" || true
git push origin ujjwal

# Create clean copy without .env for HF Spaces
git stash
CURRENT=$(git rev-parse HEAD)
git checkout --orphan _hf_temp
git rm --cached .env 2>/dev/null || true
git reset HEAD .env 2>/dev/null || true
echo ".env" >> .gitignore
git add -A
git commit -m "Deploy to HF Spaces"
git push hf _hf_temp:main --force
git checkout ujjwal
git branch -D _hf_temp
git stash pop 2>/dev/null || true

echo "Deployed to GitHub + HF Spaces!"
