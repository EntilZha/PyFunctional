
# campare_versions(v1, v2)
# Compares two 3-part sematic versions, returning -1 if v1 is less than v2, 1 if v1 is greater than v2 or 0 if v1 and v2 are equal.
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

# get_version_in_pipx(package_name)
# Gets the standard semantic version of a package installed in Pipx if installed.
get_version_in_pipx() {
  local package_name=$1
  local version
  version=$(pipx list | grep -oP "$package_name"\\s+\\K\[0-9\]+\.\[0-9\]+\.\[0-9\]+)
  echo "$version"
}

# capitalise(word)
# Capitalizes a word.
capitalize() {
  local word=$1
  echo "$(tr '[:lower:]' '[:upper:]' <<< "${word:0:1}")${word:1}"
}

# print_version(name, version, capitalize, width)
# Prints the version of the software with option to capitalize name and change left-aligned padding.
print_version() {
  local name=$1
  local version=$2
  local capitalize=${3:-true}
  local width=${4:-19}
  name=$([[ $capitalize == 'true' ]] && capitalize "$name" || echo "$name")
  printf "%-${width}s %s\n" "$name version:" "$version"
}

# install_package(package_name)
# Installs specified package with Pipx or displays the its version if it's already installed.
install_package() {
  local package_name=$1
  local capitalize=${2:-true}

  local version
  version=$(get_version_in_pipx "$package_name")
  if [[ -n $version ]]; then
    print_version "$package_name" "$version" "$capitalize"
  else
    pipx install "$package_name"
    pipx ensurepath
  fi
}

main() {
  local python_version
  python_version=$(python --version | awk '{print $2}')
  print_version "Python" "$python_version"

  local pipx_version
  pipx_version=$(pipx --version)
  if [[ -z "$pipx_version" ]]; then
    echo "Please install Pipx before running this script."
    exit 1
  else
    print_version "Pipx" "$pipx_version"
  fi

  install_package "poetry"

  install_package "pre-commit" false

  echo

  if ! poetry install; then
    poetry lock
    poetry install
  fi

  echo

  poetry run pylint functional

  echo

  poetry run ruff check functional

  echo

  poetry run black --diff --color --check functional

  echo

  poetry run mypy functional

  echo

  poetry run pytest
}

main "$@"