#!/bin/bash

# Setup script for local Jekyll development

echo "🚀 Setting up RewardAnything GitHub Pages..."

# Check if Ruby is installed
if ! command -v ruby &> /dev/null; then
    echo "❌ Ruby is not installed. Please install Ruby first."
    echo "   Visit: https://www.ruby-lang.org/en/documentation/installation/"
    exit 1
fi

# Check if Bundler is installed
if ! command -v bundle &> /dev/null; then
    echo "📦 Installing Bundler..."
    gem install bundler
fi

# Install Jekyll and dependencies
echo "📦 Installing Jekyll and dependencies..."
bundle install

echo "✅ Setup complete!"
echo ""
echo "To start the development server:"
echo "  bundle exec jekyll serve"
echo ""
echo "Then visit: http://localhost:4000/RewardAnything" 