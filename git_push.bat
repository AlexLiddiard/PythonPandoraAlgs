echo off
git config user.name "JaBCo1"
git config user.email "J.Collings@warwick.ac.uk"
git add -A
git reset -- "__pycache__/*"
git commit -m %1
git push --force origin master
