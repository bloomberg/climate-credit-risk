<!--
Before You Start

As much as possible, we have tried to provide enough tooling to get you up and
running quickly and with a minimum of effort. This includes sane defaults for
documentation; templates for bug reports, feature requests, and pull requests;
and [GitHub Actions](https://github.com/features/actions) that will
automatically manage stale issues and pull requests. This latter defaults to
labeling issues and pull requests as stale after 60 days of inactivity, and
closing them after 7 additional days of inactivity. These
[defaults](.github/workflows/stale.yml) and more can be configured. For
configuration options, please consult the documentation for the [stale
action](https://github.com/actions/stale).

In trying to keep this template as generic and reusable as possible, there are
some things that were omitted out of necessity and others that need a little
tweaking. Before you begin developing in earnest, there are a few changes that
need to be made:

- [ ] Replace `<INSERT_CONTACT_METHOD>` in
  [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) with a suitable communication
  channel.
- [ ] Change references to `org_name` to the name of the org your repository belongs
  to e.g., `bloomberg`:
  - [ ] In [`README.md`](README.md)
  - [ ] In [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [ ] Change references to `repo_name` to the name of your new repository:
  - [ ] In [`README.md`](README.md)
  - [ ] In [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [ ] Update the Release and Lint `README` badges to point to your project URL.
- [ ] Update the links to `CONTRIBUTING.md` to point to your project URL:
  - [ ] In
    [`.github/ISSUE_TEMPLATE/bug_report.yml`](.github/ISSUE_TEMPLATE/bug_report.yml)
  - [ ] In
    [`.github/ISSUE_TEMPLATE/feature_request.yml`](.github/ISSUE_TEMPLATE/feature_request.yml)
  - [ ] In
    [`.github/pull_request_template.md`](.github/pull_request_template.md)
- [ ] Update the `Affected Version` tags in
  [`.github/ISSUE_TEMPLATE/bug_report.yml`](.github/ISSUE_TEMPLATE/bug_report.yml)
  if applicable.
- [ ] Replace the `<project name>` placeholder with the name of your project:
  - [ ] In [`CONTRIBUTING.md`](CONTRIBUTING.md)
  - [ ] In [`SECURITY.md`](SECURITY.md)
- [ ] Add names and contact information for the project maintainers to
  [`MAINTAINERS.md`](MAINTAINERS.md).
- [ ] Update the `<project-name>` placeholder in
  [`.github/CODEOWNERS`](.github/CODEOWNERS) as well as the
  `<maintainer-team-name>` and `<admin-team-name>` entries.
- [ ] Delete the release placeholder content in [`CHANGELOG.md`](CHANGELOG.md).
  We encourage you to [keep a changelog](https://keepachangelog.com/en/1.0.0/).
- [ ] Configure [`.github/dependabot.yml`](.github/dependabot.yml) for your project's
  language and tooling dependencies.
- [ ] 🚨 Delete this section of the `README`!
-->
# An Efficient SSP-based Methodology for Assessing Climate Risks of a Large Credit Portfolio

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/bloomberg/climate-credit-risk/badge)](https://scorecard.dev/viewer/?uri=github.com/bloomberg/climate-credit-risk)
[![Lint](https://github.com/bloomberg/climate-credit-risk/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/marketplace/actions/super-linter)
[![Contributor-Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-fbab2c.svg)](CODE_OF_CONDUCT.md)

This repository contains the code to reproduce the results of the [paper](https://hal.science/hal-04665712/document).

## Installation

To set up the environment and install dependencies, follow these steps:

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```

3. Install the required dependencies:

   ```bash
   pip install .
   ```

After setting up the environment, you can run the scripts and notebooks in this repository to reproduce the results presented in the paper.

## Repository Structure

- **`firm.py`**: Contains the `Firm` class, which models a single firm's optimal carbon emission strategy.
- **`portfolio.ipynb`**: Jupyter Notebook for analyzing and visualizing the climate risks of a credit portfolio.
- **`utils.py`**: Utility functions and constants used across the project.

## License

Distributed under the `Apache-2.0` license. See [LICENSE](LICENSE) for more
information.
