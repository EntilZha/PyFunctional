git checkout master -- README.md
echo "---" > index.md
echo "layout: index" >> index.md
echo "---" >> index.md
cat README.md >> index.md
rm README.md

jekyll build
