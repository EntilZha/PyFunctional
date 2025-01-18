python_version=$(python --version | grep -Eo \[0-9\]\.\[0-9\]+\.\[0-9\]+)
echo "Python version: $python_version"

poetry run pytest