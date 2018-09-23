#!/bin/sh

GH_PAGES=gh_pages


rm -rf $GH_PAGES
git clone git@github.com:bbci/mushu $GH_PAGES
cd $GH_PAGES
git checkout gh-pages
rm -rf *
cp -r ../doc/_build/html/* .
touch .nojekyll

