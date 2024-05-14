#!/bin/bash
set -euo pipefail; IFS=$'\n\t'

NAME=$( python setup.py --name )
VER=$( python setup.py --version )

echo "========================================================================"
echo "Tagging $NAME v$VER"
echo "========================================================================"

git tag v$VER
git push origin v$VER

echo "========================================================================"
echo "This triggers a GitHub Actions workflow that will build wheels and"
echo "source distributions, and upload them to PyPI."
echo "========================================================================"
