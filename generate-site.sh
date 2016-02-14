rm index.html
rm -r images
rm -r javascripts
rm -r stylesheets

git checkout master -- README.md
LAYOUT_PREFIX='---\nlayout: index\n---\n\n'
echo $LAYOUT_PREFIX > jekyll-site/index.md
cat README.md >> jekyll-site/index.md
rm README.md
git rm README.md

cd jekyll-site

jekyll build
mv _site/* ..
rm -r _site
