python_version=$(python --version | grep -Eo \[0-9\]\.\[0-9\]+\.\[0-9\]+)
echo "Python version: $python_version"

pipx_version=$(pipx --version)
if [[ -z "$pipx_version" ]]; then
  echo "Pipx is not installed"
  exit 1
else
  echo "Pipx version:   $pipx_version"
fi

poetry_version=$(pipx list | grep -oP poetry\\s+\\K\[0-9\]\.\[0-9\]+\.\[0-9\]+)
if [[ -n $poetry_version ]]; then
  echo "Poetry version: $poetry_version"
else
  pipx install poetry
fi

poetry run pytest