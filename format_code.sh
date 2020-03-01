set -o xtrace

# due to symlink this will also format ./src/* too
isort --recursive ./submit/solution
black -l 112 -t py37 ./submit/solution

