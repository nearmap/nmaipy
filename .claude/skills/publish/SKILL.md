---
name: publish
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

3. **Enforce code formatting**
   - Run `black -l 120 nmaipy/ tests/` and `isort nmaipy/ tests/`
   - If any files were reformatted, stage them and present a suggested commit message for user review
   - This ensures consistent formatting before release

4. **Run test suite (including live API)**
   - Run `pytest` (all tests, including live API — requires `API_KEY` env var) and verify all tests pass
   - A release must pass the full test suite, not just the non-live subset
   - If tests fail, attempt to fix failures where appropriate (obvious bugs, missing imports, etc.)
   - Stage any fixes and present a suggested commit message for user review
   - If fixes require user decision or failures are unclear, report and ask how to proceed

5. **Code review against last release**
   - Get the diff since the last release tag: `git diff $(git describe --tags --abbrev=0)..HEAD`
   - Use the code-reviewer agent to review all changes
   - Focus on: correctness, breaking changes, security issues, missing tests
   - Report any concerns to the user and ask whether to proceed
   - This catches issues that may have bypassed proper PR review

6. **Determine release path**
   - If on a feature branch (not main/master): Create a PR to main
   - If on main/master or PR already merged: Proceed to release

7. **Build and publish to PyPI**
   - Clean old builds: `rm -rf dist/ build/ *.egg-info`
   - Build: `python -m build`
   - Verify: `twine check dist/*`
   - Upload: `twine upload dist/*`

8. **Tag and push**
   - Create annotated tag: `git tag -a vX.Y.Z -m "Release version X.Y.Z"`
   - Push tag: `git push origin vX.Y.Z`

9. **Write release notes**
   - Review all changes since last release: `git log $(git describe --tags --abbrev=0)..HEAD --oneline`
   - Write comprehensive release notes covering:
     - **What's New**: One sentence summary of the release
     - **Features**: Bullet points for new functionality (use `**bold**` for feature names)
     - **Bug Fixes**: Any bugs fixed
     - **Breaking Changes**: API or behavior changes (if any)
     - **Notes**: Any caveats, pre-release warnings, or migration guidance
   - Present draft release notes to user for review before creating the release
   - Do NOT just use `--generate-notes` alone - that only shows commit titles

10. **Create GitHub release**
   - Use `gh release create` with:
     - Tag name (vX.Y.Z)
     - Title (vX.Y.Z)
     - The release notes written in step 8 (use `--notes` with a heredoc)
     - Mark as pre-release if version contains 'a', 'b', or 'rc' (`--prerelease` flag)

11. **Update the conda-forge feedstock** (downstream of PyPI — asynchronous)

   nmaipy is also distributed on conda-forge via the feedstock at
   `https://github.com/conda-forge/nmaipy-feedstock` (recipe: `recipe/recipe.yaml`,
   `schema_version: 1` / rattler-build format). This step depends on the PyPI sdist already
   being live and does **not** complete in the same session as the PyPI publish — treat it
   as a follow-up to poll for, not a blocking step. Skip it entirely for pre-releases
   (versions containing 'a', 'b', or 'rc'): conda-forge tracks stable releases only.

   - **Wait for the autotick bot.** Within a few hours of the PyPI release the
     `regro-cf-autotick-bot` opens a version-bump PR on the feedstock that updates only
     `context.version` and `source.sha256` (a 2-line change). Find it with:
     `gh pr list --repo conda-forge/nmaipy-feedstock`.
     Do **not** upload to anaconda.org yourself — merging the feedstock PR is what makes
     conda-forge CI build the `noarch: python` package and publish it to the channel.

   - **Reconcile run dependencies — the bot does NOT do this.** This is the whole reason the
     step exists: the bot bumps version + hash only, so a dependency added, removed, or
     re-floored in nmaipy since the last conda release silently leaves the recipe wrong and
     the conda package ships with the wrong deps. Compare the two sources of truth:
       - nmaipy: `[project].dependencies` in `pyproject.toml`
       - recipe: `requirements.run` in `recipe/recipe.yaml`

     `grayskull pypi nmaipy==X.Y.Z` regenerates the run list from the published sdist
     metadata if you'd rather diff against a fresh generation than eyeball it. If they
     differ, push the fix to the bot's PR branch (the bot PR explicitly invites
     "Feel free to push to the bot's branch"):
     ```
     gh repo clone conda-forge/nmaipy-feedstock
     cd nmaipy-feedstock && git checkout <bot-branch>   # e.g. 5.0.17_h00a991, from the PR
     # edit recipe/recipe.yaml requirements.run to match pyproject deps (conda spacing: "name >=x.y")
     git commit -am "Sync run requirements with nmaipy X.Y.Z" && git push
     ```
     The recipe's `pip_check: true` test fails CI when a declared dep is missing, so a red CI
     on the bot PR usually means a `run:` dep needs adding.

   - **Merge on green CI.** You must be a listed maintainer
     (`extra.recipe-maintainers` — currently `mbewley`, `bretttully`).

   - **Manual fallback** if you can't wait for the bot: branch the feedstock, bump
     `context.version` + `source.sha256` (sha256 of the PyPI sdist tarball) and the run-deps
     in `recipe.yaml`, open a PR, merge once CI passes.

### Output

Report the final status with links to:
- PyPI package page
- GitHub release page
- conda-forge feedstock PR (note whether dependency edits were needed). This is
  asynchronous and may still be open/pending when the rest of the release is done.

**Note:** After the tag is pushed, a GitHub Actions workflow automatically runs to validate the release (runs tests, verifies version, attaches build artifacts to the release).
