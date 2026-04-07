#!/usr/bin/env bash
set -euo pipefail

REPO="/d/Users/danie/PycharmProjects/age-gender-emotion-edge"
SAFE_DIR="D:/Users/danie/PycharmProjects/age-gender-emotion-edge"

cd "$REPO"

git -c safe.directory="$SAFE_DIR" add -A
git -c safe.directory="$SAFE_DIR" \
  -c user.name="Danielkis97" \
  -c user.email="danielkis97@users.noreply.github.com" \
  commit -m "$(cat <<'EOF'
remove private report planning file from repository

Delete the IU-specific project report planning document so the repository remains general-purpose and focused on the software project artifacts only.
EOF
)"
git -c safe.directory="$SAFE_DIR" push origin HEAD
git -c safe.directory="$SAFE_DIR" status -sb
