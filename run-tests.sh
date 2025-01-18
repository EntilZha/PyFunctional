
compare_versions() {
  local v1=(${1//./ })
  local v2=(${2//./ })

  for i in {0..2}; do
    if [[ ${v1[i]} -lt ${v2[i]} ]]; then
      # Version $1 is less than $2
      echo -1
      return
    elif [[ ${v1[i]} -gt ${v2[i]} ]]; then
      # Version $1 is greater than $2"
      echo 1
      return
    fi
  done
  # "Version $1 is equal to $2"
  echo 0
}

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

if ! poetry install; then
  poetry lock
  poetry install
fi

if [[ $(compare_versions "$python_version" "3.12.0") -lt 0 ]]; then
  poetry run pylint functional
else
  poetry run ruff check functional
fi

poetry run black --check functional

poetry run mypy functional

poetry run pytest