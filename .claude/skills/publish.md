---
description: Prepare and publish a release to PyPI and GitHub
---

## Release Workflow

Execute the full release workflow for nmaipy.

### Current State

- Branch: !`git branch --show-current`
- Git status: !`git status --short`
- Current version: !`python -c "from nmaipy import __version__; print(__version__)"`
- Latest GitHub release: !`gh release list --limit 1 2>/dev/null || echo "No releases found"`
- Latest git tag: !`git describe --tags --abbrev=0 2>/dev/null || echo "No tags found"`

### Release Checklist

Execute these steps in order, stopping if any step fails:

1. **Check clean git repo**
   - Verify no uncommitted changes (staged changes are OK if they're part of the release)
   - If dirty, prompt user to commit or stash changes first

2. **Check version is new**
   - Compare current `__version__` with latest GitHub release tag
   - If version hasn't been bumped, stop and ask user to update version first

3. **Run test suite**
   - Run `pytest` and verify all tests pass
   - If tests fail, attempt to fix failures where appropriate (obvious bugs, missing imports, etc.)
   - Stage any fixes and present a suggested commit message for user review
   - If fixes require user decision or failures are unclear, report and ask how to proceed

4. **Code review against last release**
   - Get the diff since the last release tag: `git diff $(git describe --tags --abbrev=0)..HEAD`
   - Use the code-reviewer agent to review all changes
   - Focus on: correctness, breaking changes, security issues, missing tests
   - Report any concerns to the user and ask whether to proceed
   - This catches issues that may have bypassed proper PR review

5. **Determine release path**
   - If on a feature branch (not main/master): Create a PR to main
   - If on main/master or PR already merged: Proceed to release

6. **Build and publish to PyPI**
   - Clean old builds: `rm -rf dist/ build/ *.egg-info`
   - Build: `python -m build`
   - Verify: `twine check dist/*`
   - Upload: `twine upload dist/*`

7. **Tag and push**
   - Create annotated tag: `git tag -a vX.Y.Z -m "Release version X.Y.Z"`
   - Push tag: `git push origin vX.Y.Z`

8. **Create GitHub release**
   - Use `gh release create` with:
     - Tag name (vX.Y.Z)
     - Title (vX.Y.Z)
     - Auto-generated release notes from commits since last release
     - Mark as pre-release if version contains 'a', 'b', or 'rc'

### Output

Report the final status with links to:
- PyPI package page
- GitHub release page

**Note:** After the tag is pushed, a GitHub Actions workflow automatically runs to validate the release (runs tests, verifies version, attaches build artifacts to the release).
