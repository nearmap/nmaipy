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
   - `build` runs the offline tests, reads `__version__`, and fails if the tag `vX.Y.Z` exists at another commit or PyPI already has the version. For a stable version, it checks that the conda-forge recipe can build the version. Then it builds the sdist and the wheel with `pixi run build`.
   - `check-wheel` installs the built wheel into a clean virtual environment and checks it, before `publish` starts.
   - `publish` waits for the approval, tags the run's commit `vX.Y.Z` and publishes the sdist and the wheel to PyPI through trusted publishing, with attestations.
   - `github-release` creates the GitHub release `vX.Y.Z` with generated notes and the same two files, marked as a pre-release for an `a`, `b` or `rc` version, and fails unless the release's assets have the same SHA-256 digests as the files.
5. Replace the generated notes with `gh release edit vX.Y.Z --notes-file <file>`. The tag's Memory benchmark run (`.github/workflows/memory-benchmark.yml`) adds a "Memory benchmark" section to the notes when it finishes, and `--notes-file` replaces the whole body, so write the notes before that run finishes, or re-run that run's `notes` job after any `--notes-file` edit (the merge is idempotent). A run's jobs can be re-run only for 30 days; after that, run `gh workflow run memory-benchmark.yml --ref vX.Y.Z`, which measures again (up to 5 hours) and replaces the section. Write them from `git log <previous tag>..vX.Y.Z --oneline`, with this outline:
   - **What's New**: One sentence summary of the release
   - **Features**: Bullet points for new functionality (use `**bold**` for feature names)
   - **Bug Fixes**: Any bugs fixed
   - **Breaking Changes**: API or behavior changes (if any)
   - **Notes**: Any caveats, pre-release warnings, or migration guidance

   The notes stay editable on an immutable release. Its assets and its tag do not.

By policy, only the Release workflow publishes to PyPI or creates tags. It also creates each version's GitHub release, which is immutable and has the PyPI files.

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
- If `build` fails at the conda-forge recipe check, open the feedstock pull request that [conda-forge feedstock](#conda-forge-feedstock) describes, or merge it if it is open. Wait a few minutes after the merge, because the check reads the recipe through a cache, then run the workflow again. If the message names `pyproject.toml`, the fix is in `pyproject.toml` or in `scripts/check_feedstock_recipe.py`, not in a feedstock pull request.
- A version that can never be published leaves a tag that only someone who can bypass the tag ruleset (for example, a repository admin) can remove. Bump to the next version and release that instead.

## conda-forge feedstock

nmaipy is also on conda-forge, from the feedstock at `https://github.com/conda-forge/nmaipy-feedstock`. The recipe is `recipe/recipe.yaml`.

Within a day, usually 1 to 2 hours after a stable version is on PyPI, `regro-cf-autotick-bot` opens the pull request `[bot-automerge] nmaipy vX.Y.Z` on the feedstock. It updates `context.version` and `source.sha256`, and it tries to update `requirements.run` from the sdist (add and remove run dependencies, change their floors). That analysis is best effort: when it fails, the bot says so in the "Dependency Analysis" section of its pull request, and the pull request still merges itself. It merges the pull request itself when the `linter` and `github-actions` checks pass. A pre-release gets no pull request.

Do not open a version pull request by hand. Once the version is on the feedstock, the bot opens no pull request, so a pull request by hand repeats the bot's work and skips its update of the run dependencies. If no bot pull request appears within a day, check https://conda-forge.org/status/#version_updates, then open an issue on the feedstock titled `@conda-forge-admin, please update version`.

The bot never edits `requirements.host` or `context.python_min`. When a change to `[build-system].requires` or `requires-python` in `pyproject.toml` is merged, open a feedstock pull request before the next release. It adds the new build backend to `requirements.host` beside the old one, sets `context.python_min` to the new floor, and bumps `build.number`. The `build` job of the Release workflow refuses a stable release until the recipe matches.

If a bot pull request fails, push the fix to the bot's branch, then add the `automerge` label: a commit by anyone but the bot stops the automatic merge.

Merging a feedstock pull request needs a recipe maintainer (`extra.recipe-maintainers` in the recipe).
