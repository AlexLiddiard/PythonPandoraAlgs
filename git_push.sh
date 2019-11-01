#!/bin/bash
git add -A
git reset -- "Python Code/__pycache__/*"
git commit -m "$1"
git push --force origin master
