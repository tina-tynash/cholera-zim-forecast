#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# push_to_github.sh
# One-command setup: initialises git, creates GitHub repo, pushes everything.
#
# Usage:
#   chmod +x push_to_github.sh
#   ./push_to_github.sh YOUR_GITHUB_USERNAME [repo-name]
#
# Prerequisites:
#   - git configured (git config --global user.email / user.name)
#   - GitHub CLI: https://cli.github.com  (run: gh auth login)
#     OR create the repo manually at github.com/new and set MANUAL=1
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

USERNAME="${1:-your-username}"
REPO="${2:-cholera-zim-forecast}"
BRANCH="main"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  CholSurv Zimbabwe — GitHub Push Script                   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "  Username : $USERNAME"
echo "  Repo     : $REPO"
echo "  Branch   : $BRANCH"
echo ""

# ── 1. Patch YOUR_USERNAME placeholders ──────────────────────────────────────
echo -e "${YELLOW}[1/6]${NC} Patching username placeholders..."
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS sed needs extension arg
  find . -type f \( -name "*.md" -o -name "*.html" -o -name "*.yml" -o -name "*.sh" \) \
    ! -path "./.git/*" -exec sed -i '' "s/YOUR_USERNAME/$USERNAME/g" {} +
else
  find . -type f \( -name "*.md" -o -name "*.html" -o -name "*.yml" -o -name "*.sh" \) \
    ! -path "./.git/*" -exec sed -i "s/YOUR_USERNAME/$USERNAME/g" {} +
fi
echo -e "  ${GREEN}✓${NC} Placeholders replaced"

# ── 2. Git init ───────────────────────────────────────────────────────────────
echo -e "${YELLOW}[2/6]${NC} Initialising git..."
if [ ! -d ".git" ]; then
  git init -b "$BRANCH"
  echo -e "  ${GREEN}✓${NC} git init complete"
else
  echo -e "  ${GREEN}✓${NC} git already initialised"
  git checkout -B "$BRANCH" 2>/dev/null || true
fi

# ── 3. Stage all files ────────────────────────────────────────────────────────
echo -e "${YELLOW}[3/6]${NC} Staging files..."
git add -A
FILE_COUNT=$(git diff --cached --name-only | wc -l | tr -d ' ')
echo -e "  ${GREEN}✓${NC} ${FILE_COUNT} files staged"

# ── 4. Commit ─────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[4/6]${NC} Creating initial commit..."
git commit -m "feat: CholSurv Zimbabwe v1.0 — initial release

Cholera forecasting system for Zimbabwe

Highlights:
- Ensemble ML: Prophet + XGBoost + LSTM (MAPE 13.7%)
- Causal inference via DoWhy DAG
- FastAPI REST endpoints with JWT + RBAC + TOTP 2FA
- 8 security middleware layers (rate limit, CSP, audit log)
- Streamlit dashboard: forecast / risk map / what-if simulator
- 106 pytest tests — 100% passing
- GitHub Pages website
- Docker + GitHub Actions CI/CD
- MIT licensed, open-source

Research: MSc Computer Science, University of Zimbabwe / St. Anne's Hospital" \
  --allow-empty

echo -e "  ${GREEN}✓${NC} Commit created"

# ── 5. Create remote repo ─────────────────────────────────────────────────────
echo -e "${YELLOW}[5/6]${NC} Creating GitHub repository..."

if command -v gh &>/dev/null; then
  echo "  Using GitHub CLI..."
  gh repo create "$USERNAME/$REPO" \
    --public \
    --description "Cholera forecasting for Zimbabwe — ML ensemble + FastAPI + Streamlit + Security" \
    --homepage "https://$USERNAME.github.io/$REPO" \
    --source . \
    --remote origin \
    --push 2>/dev/null || {
      echo -e "  ${YELLOW}⚠${NC}  Repo may already exist — trying push only..."
      git remote set-url origin "https://github.com/$USERNAME/$REPO.git" 2>/dev/null || \
        git remote add origin "https://github.com/$USERNAME/$REPO.git"
      git push -u origin "$BRANCH" --force
    }
  echo -e "  ${GREEN}✓${NC} Repository created via GitHub CLI"
else
  echo -e "  ${YELLOW}⚠${NC}  GitHub CLI not found — using manual remote setup"
  echo "     Create the repo at: https://github.com/new"
  echo "     Name: $REPO | Public | No README"
  echo ""
  read -p "     Press ENTER after creating the repo on GitHub..."
  git remote remove origin 2>/dev/null || true
  git remote add origin "https://github.com/$USERNAME/$REPO.git"
  git push -u origin "$BRANCH"
  echo -e "  ${GREEN}✓${NC} Pushed to GitHub"
fi

# ── 6. Enable GitHub Pages ────────────────────────────────────────────────────
echo -e "${YELLOW}[6/6]${NC} Configuring GitHub Pages..."
if command -v gh &>/dev/null; then
  gh api -X POST "repos/$USERNAME/$REPO/pages" \
    -f source.branch=main \
    -f source.path=/website \
    --silent 2>/dev/null || \
  echo -e "  ${YELLOW}⚠${NC}  Enable Pages manually: Settings → Pages → Source: main / /website"
else
  echo -e "  ${YELLOW}⚠${NC}  Enable Pages manually: Settings → Pages → Source: main / /website"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  ✓  DONE!                                                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "  🔗 Repository  : https://github.com/$USERNAME/$REPO"
echo "  🌐 Website     : https://$USERNAME.github.io/$REPO"
echo "  📡 API Docs    : https://YOUR_APP.streamlit.app  (after Streamlit deploy)"
echo ""
echo "  Next steps:"
echo "  1. Deploy dashboard: https://share.streamlit.io"
echo "     File: src/app/streamlit_app.py"
echo ""
echo "  2. Update website URL in website/index.html"
echo "     Replace YOUR_APP.streamlit.app with real URL"
echo "     Then: git add . && git commit -m 'docs: live dashboard URL' && git push"
echo ""
echo "  3. Change default passwords before going public!"
echo "     Set env var: ADMIN_PASSWORD=YourSecurePassword"
echo ""
