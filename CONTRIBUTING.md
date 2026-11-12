# Contributing to nmaipy

Anyone can open an [issue](https://github.com/nearmap/nmaipy/issues) or a pull request.

## Pull requests

If you do not have write access to `nearmap/nmaipy` (you are outside Nearmap, or at Nearmap without write access), fork the repository and open a pull request against `main`. Nearmap engineers with write access push a branch to `nearmap/nmaipy` and open the pull request from it.

A pull request merges only when a code owner (`@nearmap/ai-systems`, from `.github/CODEOWNERS`) has approved it, all review threads are resolved, the required checks pass and every commit is signed. Only members of `@nearmap/ai-systems` and repository admins can merge.

## Continuous integration

Every pull request runs these workflows from `.github/workflows/`:

- Tests (`test.yml`): `pixi run test` in each test environment, and a check that the built wheel installs with pip into a clean virtual environment, imports, and carries every `.py` and `.json` file of the package (`scripts/check_wheel.py`).
- Lint (`lint.yml`): `pixi run lint`.
- Docs (`docs.yml`): `pixi run docs`, when the pull request changes `docs/`, `README.md`, `pyproject.toml`, `pixi.lock` or the workflow file.

Pull request CI does not run the live API tests.

For an author outside the Nearmap GitHub organization, a maintainer approves the workflow runs before they start.

## Development setup

You need [pixi](https://pixi.sh) 0.83 or newer (`requires-pixi` in `pyproject.toml`). On Linux or macOS, install it with:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

On Windows, install it with:

```powershell
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

The workspace solves only for `linux-64`, `osx-arm64` and `win-64` (`platforms` under `[tool.pixi.workspace]` in `pyproject.toml`).

Open a new shell, then clone the repository (or your fork), create the environment and run the tests:

```bash
git clone https://github.com/nearmap/nmaipy.git
cd nmaipy
pixi install
pixi run test
```

## Checks

- `pixi run test` runs the offline test suite (`pytest -m 'not live_api'`).
- `pixi run lint` runs ty, then `ruff check` and `ruff format --check`. This is the same check as the Lint workflow.
- `pixi run format` applies the fixes that ruff can make safely, then formats the code.
- `pixi run docs` builds the documentation site. Run it after a change to `docs/` or `README.md`.
- `pixi run -e <environment> test` runs the tests in another environment, for another Python version. The environments are under `[tool.pixi.environments]` in `pyproject.toml`.

The live API tests need an `API_KEY` environment variable and run with `pixi run pytest`.

The architecture and the code conventions are in `CLAUDE.md`.

## Signed commits

Every commit on the pull request branch needs a verified signature, or the pull request cannot merge. This holds for a squash merge too.

To sign your commits with an SSH key:

```bash
git config --global gpg.format ssh
git config --global user.signingkey <path to the .pub key>
git config --global commit.gpgsign true
```

Then add the key on GitHub as a signing key, and commit with an email address (`user.email`) that is verified on that account. For GPG keys and the other options, see [GitHub's documentation on commit signature verification](https://docs.github.com/en/authentication/managing-commit-signature-verification).

To sign every commit of an existing branch without moving its base, and keep any merge commits on it, run this on the branch:

```bash
git fetch https://github.com/nearmap/nmaipy.git main
before=$(git rev-parse HEAD)
git rebase --rebase-merges --force-rebase --gpg-sign "$(git merge-base HEAD FETCH_HEAD)"
git diff --stat --exit-code "$before" HEAD && git push --force-with-lease
```

If the rebase stops on a conflict, resolve it and run `git rebase --continue` until the rebase finishes. Then run the last line again, not the whole block, which resets `before`. If the last line lists files instead of pushing, the rebase dropped an edit that was made in a merge commit: re-apply it (`git diff HEAD "$before"` shows it), amend, and run the last line again.

## Public content

nmaipy is a public, standalone package. Everything that becomes part of the repository or its history (code, comments, docstrings, log strings, test data, commit messages, PR titles and descriptions, review comments, release notes) must stand on its own:

- Do not reference specific applications, pipelines, deployment tooling or other harnesses that consume nmaipy, by name or by description. State the need in terms of nmaipy's own API ("callers of `AOIExporter` need to widen the retry budget"), not who the caller is.
- Do not include internal ticket ids, project or dataset names, customer or partner names, AOI/parcel counts or regions from a known workload, storage paths, or anything else that identifies an internal or commercial use of the package. Describe the issue generically ("a large export with spatially sorted input over a coverage gap").
- Tests use synthetic or bundled example data, never excerpts of production outputs.
- When editing an existing PR's description or replying to a review, the same rules apply.

If a change only makes sense in the context of a downstream system, explain it in terms of nmaipy's behaviour and leave the downstream motivation out.

## License

nmaipy is licensed under the Apache License 2.0 (`LICENSE`). Under section 5 of the license, a contribution that you submit for inclusion is under the Apache License 2.0, unless you state otherwise. There is no contributor license agreement and no sign-off line.
