# PyPI Trusted Publisher Setup Guide

This repository uses PyPI Trusted Publishers for secure, automated publishing.

## What is Trusted Publishing?

Trusted publishing uses OpenID Connect (OIDC) to allow GitHub Actions to publish directly to PyPI without API tokens. This is more secure and is the recommended approach by PyPI.

## Setup Instructions

### Step 1: Configure PyPI Trusted Publisher

**For the first release (pending publisher):**

1. Go to: https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in the form with these exact values:

```
PyPI Project Name: pyclnf
GitHub Repository Owner: johnwilsoniv
Repository Name: pyclnf
Workflow Name: publish.yml
Environment Name: release
```

4. Click "Add"

**After the first successful publish:**

The pending publisher will automatically convert to a regular trusted publisher. For subsequent configuration:

1. Go to: https://pypi.org/project/pyclnf/
2. Navigate to "Settings" → "Publishing"
3. Manage trusted publishers there

### Step 2: Create GitHub Environments (Optional but Recommended)

For extra security with manual approval:

1. Go to: https://github.com/johnwilsoniv/pyclnf/settings/environments
2. Create environment named `release`:
   - Add yourself as required reviewer
   - Restrict to `main` branch only
3. Create environment named `test-release` (for TestPyPI testing):
   - No special restrictions needed

### Step 3: Configure TestPyPI (Optional)

For testing before production release:

1. Go to: https://test.pypi.org/manage/account/publishing/
2. Add pending publisher with same details as above

## How to Publish

### Automatic Publishing (Recommended)

Publishing happens automatically when you create a GitHub release:

```bash
# Tag a new version
git tag v0.2.0
git push origin v0.2.0

# Create GitHub release
gh release create v0.2.0 \
  --title "PyCLNF v0.2.0" \
  --notes "Release notes here"
```

The workflow will:
1. Build the package
2. Publish to PyPI (requires environment approval if configured)

### Manual Testing with TestPyPI

To test the workflow without publishing to production PyPI:

1. Go to: https://github.com/johnwilsoniv/pyclnf/actions/workflows/publish.yml
2. Click "Run workflow"
3. Select branch and click "Run workflow"
4. This will publish to TestPyPI only

## Workflow Details

The workflow (`.github/workflows/publish.yml`) consists of three jobs:

1. **build**: Builds the distribution packages
2. **publish-to-pypi**: Publishes to production PyPI (on release)
3. **publish-to-testpypi**: Publishes to TestPyPI (on manual trigger only)

## Security Features

- **No API tokens stored**: Uses OIDC authentication
- **Environment protection**: Requires manual approval before publishing
- **Branch restrictions**: Only deploys from `main` branch
- **Audit trail**: All publishes tracked in GitHub Actions logs

## Troubleshooting

### "Trusted publisher not configured"

Make sure you've added the pending publisher on PyPI before the first release.

### "Environment not found"

Create the `release` environment in GitHub repository settings.

### "Permission denied"

Ensure the workflow has `id-token: write` permission (already configured).

## Current Configuration

- **Repository**: https://github.com/johnwilsoniv/pyclnf
- **Workflow**: `.github/workflows/publish.yml`
- **Environments**: `release` (PyPI), `test-release` (TestPyPI)
- **PyPI Project**: https://pypi.org/project/pyclnf/

## Alternative: Manual Publishing

If you prefer manual publishing, you can still use API tokens:

```bash
# Generate token at: https://pypi.org/manage/account/token/
# Then:
twine upload dist/*
```

But trusted publishing is more secure and recommended.
