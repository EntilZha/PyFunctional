git checkout master -- README.md
LAYOUT_PREFIX='---\nlayout: index\n---\n\n'
echo $LAYOUT_PREFIX > index.md
cat README.md >> index.md
rm README.md

jekyll build
