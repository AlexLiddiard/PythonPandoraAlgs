git add -A
git reset -- "__pycache__/*"
git commit -m "$1"
git push --force origin master
