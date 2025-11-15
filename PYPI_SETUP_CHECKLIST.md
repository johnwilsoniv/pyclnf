# PyPI Trusted Publisher Setup - Quick Checklist

## Step 1: Configure PyPI Trusted Publisher

### Go to PyPI Pending Publishers Page
**URL**: https://pypi.org/manage/account/publishing/

### Fill in the Form (Exact Values)

```
┌─────────────────────────────────────────────────┐
│ Add a new pending publisher                     │
├─────────────────────────────────────────────────┤
│ PyPI Project Name:     pyclnf                   │
│ GitHub Owner:          johnwilsoniv             │
│ Repository Name:       pyclnf                   │
│ Workflow Name:         publish.yml              │
│ Environment Name:      release                  │
└─────────────────────────────────────────────────┘
```

Click **"Add"**

---

## Step 2: (Optional) Set Up TestPyPI

### Go to TestPyPI Pending Publishers Page
**URL**: https://test.pypi.org/manage/account/publishing/

### Fill in the Same Form

Same values as above, then click **"Add"**

---

## Step 3: (Optional) Create GitHub Environments

### Go to GitHub Environments
**URL**: https://github.com/johnwilsoniv/pyclnf/settings/environments

### Create "release" Environment
1. Click "New environment"
2. Name: `release`
3. Add protection rules:
   - ✓ Required reviewers: johnwilsoniv
   - ✓ Deployment branches: Selected branches → `main`
4. Click "Save protection rules"

### Create "test-release" Environment
1. Click "New environment"
2. Name: `test-release`
3. No protection rules needed
4. Click "Configure environment"

---

## How to Publish

### First Publication (v0.1.0 already exists)

The v0.1.0 release already exists. To re-trigger publication:

```bash
# Option 1: Manual workflow trigger (TestPyPI)
# Go to: https://github.com/johnwilsoniv/pyclnf/actions/workflows/publish.yml
# Click "Run workflow" → Select "main" → "Run workflow"

# Option 2: Create a new patch release
git tag v0.1.1
git push origin v0.1.1
gh release create v0.1.1 --title "PyCLNF v0.1.1" --notes "Patch release"
```

### Subsequent Releases

```bash
# Tag new version
git tag v0.2.0
git push origin v0.2.0

# Create release (triggers workflow automatically)
gh release create v0.2.0 \
  --title "PyCLNF v0.2.0" \
  --notes "Release notes here"
```

The workflow will automatically:
1. Build package
2. Wait for your approval (if environment protection enabled)
3. Publish to PyPI

---

## Verification

After setup, verify the configuration:

1. **PyPI Pending Publisher**: https://pypi.org/manage/account/publishing/
   - Should show pyclnf pending publisher

2. **GitHub Actions**: https://github.com/johnwilsoniv/pyclnf/actions
   - Workflow should be visible

3. **Test Run**: Manually trigger workflow to test

---

## Current Status

- ✅ GitHub repository created
- ✅ Release v0.1.0 tagged
- ✅ GitHub Actions workflow configured
- ✅ Documentation complete
- ⏳ **Next**: Configure PyPI trusted publisher
- ⏳ **Then**: Trigger workflow to publish

---

## Support Links

- **PyPI Account**: https://pypi.org/account/login/
- **PyPI Pending Publishers**: https://pypi.org/manage/account/publishing/
- **TestPyPI Account**: https://test.pypi.org/account/login/
- **GitHub Actions**: https://github.com/johnwilsoniv/pyclnf/actions
- **GitHub Environments**: https://github.com/johnwilsoniv/pyclnf/settings/environments
- **Documentation**: TRUSTED_PUBLISHER_SETUP.md

---

## Alternative: Manual Upload (Fallback)

If you prefer to upload manually this time:

```bash
# Install twine
pip install twine

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Then upload to PyPI
twine upload dist/*
```

You'll need API tokens from:
- PyPI: https://pypi.org/manage/account/token/
- TestPyPI: https://test.pypi.org/manage/account/token/
