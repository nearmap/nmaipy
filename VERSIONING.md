# Version Management & Release

## Version numbers

Run `bump_version.py`, which updates both copies of the version: `nmaipy/__version__.py` and the `[tool.pixi.package]` version in `pyproject.toml`:
```bash
python bump_version.py patch   # or minor, major, or a specific version such as 4.0.1
```

**Semantic Versioning:**
- MAJOR: Breaking API changes
- MINOR: New capability, backwards compatible — something users could not do before
- PATCH: Bug fixes, plus small additive changes that add no new capability

Judge MINOR vs PATCH on *capability*, not on diff shape or commit type. A change is a PATCH if users could already achieve the same result before it landed and the change only makes it easier. Registering a friendly alias for a dataset that was already reachable by raw resource UUID is a PATCH, not a MINOR — a new constant is not a new feature.

**Pre-release Versions (PEP 440):**
- Alpha: `4.0.0a1` - early testing, API may change
- Beta: `4.0.0b1` - feature complete, testing for bugs
- Release Candidate: `4.0.0rc1` - final testing

## Before a release

Run the full test suite, including the live API tests, with `API_KEY` set: `pixi run pytest`. Pull request CI runs only the offline tests.

Review every change since the last release: `git fetch origin && git diff $(git describe --tags --abbrev=0 origin/main)..origin/main`.

## Release

1. On a branch, bump the version with `bump_version.py`, open a pull request and merge it.
2. Check that the trusted publisher exists on pypi.org (see "Why the release is trusted"): without it the `publish` job tags the commit and then fails at the upload, which leaves a stray tag. Run the Release workflow (`.github/workflows/release.yml`) from `main`: `gh workflow run release.yml --ref main`, or "Run workflow" on the workflow's page in the Actions tab.
3. A member of `@nearmap/ai-systems` approves the run's deployment to the `pypi` environment, only when CI is green on `main` at the commit that the run shows. The environment accepts only runs from `main`.
4. The workflow runs four jobs:
   - `build` runs the offline tests, reads `__version__`, and fails if the tag `vX.Y.Z` exists at another commit or PyPI already has the version. Then it builds the sdist and the wheel with `pixi run build`.
   - `check-wheel` installs the built wheel into a clean virtual environment and checks it, before `publish` starts.
   - `publish` waits for the approval, tags the run's commit `vX.Y.Z` and publishes the sdist and the wheel to PyPI through trusted publishing, with attestations.
   - `github-release` creates the GitHub release `vX.Y.Z` with generated notes and the same two files, marked as a pre-release for an `a`, `b` or `rc` version, and fails unless the release's assets have the same SHA-256 digests as the files.
5. Replace the generated notes with `gh release edit vX.Y.Z --notes-file <file>`. Write them from `git log <previous tag>..vX.Y.Z --oneline`, with this outline:
   - **What's New**: One sentence summary of the release
   - **Features**: Bullet points for new functionality (use `**bold**` for feature names)
   - **Bug Fixes**: Any bugs fixed
   - **Breaking Changes**: API or behavior changes (if any)
   - **Notes**: Any caveats, pre-release warnings, or migration guidance

   The notes stay editable on an immutable release. Its assets and its tag do not.

Only the Release workflow publishes to PyPI, and by policy it is the only thing that creates tags. It also creates each version's GitHub release, which is immutable and has the PyPI files.

### Why the release is trusted

- The `publish` job has `id-token: write`, so GitHub gives it a short-lived OIDC token that names the repository, the workflow file and the environment. PyPI exchanges the token for an upload token only when those names match the trusted publisher that a PyPI owner of nmaipy adds on pypi.org (Manage, Publishing): `nearmap`, `nmaipy`, `release.yml`, `pypi`. Do not run the workflow for a new version until that publisher exists. The workflow holds no PyPI token.
- PyPI does not check the branch. The `pypi` environment's rule that accepts only `main`, and its required approval, are what hold.
- The tag is pushed with a deploy key that the tag ruleset lets bypass. The ruleset's `DeployKey` bypass has no actor id, so it covers every deploy key of the repository: keep one deploy key.
- `RELEASE_DEPLOY_KEY` is a secret of the `pypi` environment. Keep it there. As a repository secret, any workflow on any branch could read it.

## A failed run

- If the run fails in `build` or `check-wheel`, nothing is tagged or published. Fix the cause in a pull request and run the workflow again.
- If the run fails in `publish`, re-run the failed jobs ("Re-run failed jobs" on the run's page, or `gh run rerun <run id> --failed`), and do not start a new run: once PyPI has a file of the version, `build` refuses it. The re-run needs a new approval. Its tag step accepts the tag that an earlier attempt of the run pushed, and PyPI accepts the identical files again.
- If the run fails in `github-release` at the release creation, re-run that job. `gh release create` deletes its own draft release when it fails. If it fails at the asset check, the release exists and its files differ from the ones published to PyPI: ask a repository admin, and do not re-run the job.
- If `publish` fails with `invalid-publisher`, the trusted publisher on pypi.org is missing or no longer matches the repository, workflow file name or environment name (for example after a rename). A re-run does not fix it: a PyPI owner of nmaipy corrects the publisher, and then someone re-runs the failed jobs. The tag from the failed attempt stays and the re-run accepts it.
- A version that can never be published leaves a tag that only someone who can bypass the tag ruleset (a repository admin) can remove. Bump to the next version and release that instead.

## conda-forge feedstock

nmaipy is also distributed on conda-forge via the feedstock at `https://github.com/conda-forge/nmaipy-feedstock` (recipe: `recipe/recipe.yaml`, `schema_version: 1` / rattler-build format). This step depends on the PyPI sdist already being live and does **not** complete in the same session as the PyPI publish — treat it as a follow-up to poll for, not a blocking step. Skip it entirely for pre-releases (versions containing 'a', 'b', or 'rc'): conda-forge tracks stable releases only.

- **Wait for the autotick bot.** Within a few hours of the PyPI release the `regro-cf-autotick-bot` opens a version-bump PR on the feedstock that updates only `context.version` and `source.sha256` (a 2-line change). Find it with: `gh pr list --repo conda-forge/nmaipy-feedstock`. Do **not** upload to anaconda.org yourself — merging the feedstock PR is what makes conda-forge CI build the `noarch: python` package and publish it to the channel.

- **Reconcile run dependencies — the bot does NOT do this.** This is the whole reason the step exists: the bot bumps version + hash only, so a dependency added, removed, or re-floored in nmaipy since the last conda release silently leaves the recipe wrong and the conda package ships with the wrong deps. Compare the two sources of truth:
    - nmaipy: `[project].dependencies` in `pyproject.toml`
    - recipe: `requirements.run` in `recipe/recipe.yaml`

  `grayskull pypi nmaipy==X.Y.Z` regenerates the run list from the published sdist metadata if you'd rather diff against a fresh generation than eyeball it. If they differ, push the fix to the bot's PR branch (the bot PR explicitly invites "Feel free to push to the bot's branch"):
  ```
  gh repo clone conda-forge/nmaipy-feedstock
  cd nmaipy-feedstock && git checkout <bot-branch>   # e.g. 5.0.17_h00a991, from the PR
  # edit recipe/recipe.yaml requirements.run to match pyproject deps (conda spacing: "name >=x.y")
  git commit -am "Sync run requirements with nmaipy X.Y.Z" && git push
  ```
  The recipe's `pip_check: true` test fails CI when a declared dep is missing, so a red CI on the bot PR usually means a `run:` dep needs adding.

- **Reconcile the build backend.** nmaipy builds with hatchling. If the recipe's `requirements.host` still lists `setuptools` and `wheel`, replace them with `hatchling >=1.27` in the same push.

- **Reconcile the Python floor — the bot does NOT do this either.** If `requires-python` in `pyproject.toml` moved since the last conda release, set `context.python_min` in `recipe/recipe.yaml` to its floor (e.g. `"3.13"`) in the same push.

- **Merge on green CI.** You must be a listed maintainer (`extra.recipe-maintainers` — currently `mbewley`, `bretttully`).

- **Manual fallback** if you can't wait for the bot: branch the feedstock, bump `context.version` + `source.sha256` (sha256 of the PyPI sdist tarball), then apply the three reconciliations above to the same `recipe.yaml`: run deps, the `hatchling` build backend and `context.python_min`. Open a PR, merge once CI passes.
