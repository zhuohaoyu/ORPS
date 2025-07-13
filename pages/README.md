# RewardAnything GitHub Pages

This directory contains the GitHub Pages website for the RewardAnything project.

## 🏗️ Structure

```
pages/
├── _config.yml              # Jekyll configuration
├── _layouts/
│   └── default.html         # Main layout template
├── index.html               # Homepage content
├── assets/
│   ├── images/              # Logo and image placeholders
│   └── favicon.svg          # Site favicon
├── Gemfile                  # Ruby dependencies
├── setup.sh                 # Local setup script
└── README.md               # This file
```

## 🚀 Automatic Deployment

The site automatically deploys to `https://zhuohaoyu.github.io/RewardAnything` whenever:

1. **Changes are pushed** to the `main` branch in the `pages/` directory
2. **Manual trigger** via GitHub Actions tab

The deployment is handled by the GitHub Actions workflow in `.github/workflows/deploy-pages.yml`.

## 🏠 Local Development

### Quick Setup

```bash
# Navigate to pages directory
cd pages

# Run setup script (macOS/Linux)
chmod +x setup.sh
./setup.sh

# Start development server
bundle exec jekyll serve
```

### Manual Setup

```bash
# Install Ruby dependencies
gem install jekyll bundler
bundle install

# Serve the site locally
bundle exec jekyll serve --livereload
```

Then visit: `http://localhost:4000/RewardAnything`

## 📝 Configuration

### GitHub Pages Settings

1. Go to **Repository Settings** → **Pages**
2. Source: **GitHub Actions**
3. The workflow will handle the rest automatically

### Environment Variables

The following are configured in `_config.yml`:

- `github_username`: Your GitHub username
- `paper_url`: Link to your arXiv paper
- `huggingface_url`: Link to model weights
- `pypi_url`: Link to PyPI package

## 🎨 Customization

### Replacing Placeholder Images

Replace the SVG placeholders in `assets/images/` with your actual logos:

- `logo-placeholder.svg` → Navigation logo
- `logo-placeholder-white.svg` → Footer logo (white version)  
- `hero-logo-placeholder.svg` → Large hero section logo
- `favicon.svg` → Browser favicon

### Updating Content

- **Homepage**: Edit `index.html`
- **Navigation**: Modify `_layouts/default.html`
- **Site settings**: Update `_config.yml`
- **Styling**: Customize Tailwind classes in templates

### Adding New Pages

Create new `.html` or `.md` files with front matter:

```yaml
---
layout: default
title: "Page Title"
description: "Page description"
---

Your content here...
```

## 🔧 Troubleshooting

### Local Development Issues

```bash
# Clean build files
bundle exec jekyll clean

# Rebuild dependencies
bundle install --force

# Verbose build for debugging
bundle exec jekyll serve --verbose
```

### Deployment Issues

1. Check **Actions** tab for build logs
2. Ensure `pages/` directory changes are pushed to `main`
3. Verify GitHub Pages settings are correct

## 📊 Performance

The site is optimized for:
- ✅ Mobile responsiveness
- ✅ Fast loading (Tailwind CSS via CDN)
- ✅ SEO optimization
- ✅ Accessibility
- ✅ Modern browsers

## 🤝 Contributing

When making changes:

1. Test locally first: `bundle exec jekyll serve`
2. Commit changes to `pages/` directory
3. Push to `main` branch
4. Automatic deployment will trigger

---

**Live Site**: https://zhuohaoyu.github.io/RewardAnything 