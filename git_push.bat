echo off
git add -A
git reset -- "__pycache__/*"
git commit -m %arg1%
git push --force origin master
