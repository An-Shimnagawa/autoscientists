#!/bin/bash
# Clone karpathy/autoresearch into task-autoresearch/repo/ so the AutoScientists
# loop has a train.py to evolve. Run once before the first launch.
#
# Re-running is a no-op (will skip if repo/ already exists).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

UPSTREAM="https://github.com/karpathy/autoresearch.git"
TARGET="repo"

if [ -d "$TARGET/.git" ]; then
  echo "$TARGET/ already exists. To refresh: rm -rf $SCRIPT_DIR/$TARGET && rerun this script."
  exit 0
fi

echo "Cloning $UPSTREAM → $SCRIPT_DIR/$TARGET (shallow, latest commit only) ..."
# --depth=1 fetches only the latest commit (no history) — we evolve train.py
# locally; upstream history isn't useful. Keeps .git compact.
git clone --depth=1 "$UPSTREAM" "$TARGET"

echo ""
echo "Done. $SCRIPT_DIR/$TARGET now contains the upstream repo."
echo "Follow $TARGET/README.md for any additional data-prep steps before launching."
