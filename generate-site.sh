rm index.html
rm -r images
rm -r javascripts
rm -r stylesheets

git checkout master -- README.md
LAYOUT_PREFIX='---\r\nlayout: index\r\n---\r\n\r\n'
echo $LAYOUT_PREFIX > jekyll-site/index.md
cat README.md >> jekyll-site/index.md
rm README.md

cd jekyll-site

jekyll build
mv _site/* ..
rm -r _site
