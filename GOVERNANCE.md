# Governance

## Independence

WakeWordWorld exists to compare wake word engines fairly. To keep that credible:

- Maintainers do not develop, sell or consult for a wake word engine while maintaining
  this project. A maintainer who starts to must step down from result publication.
- The project accepts no funding or in-kind support that is conditional on results,
  ranking, inclusion or exclusion of any engine.
- Engine vendors may contribute adapters, configurations and documentation. They do not
  run the evaluations that appear on the leaderboard, and they do not review their own
  submissions.
- All disagreements about configuration or results are handled in public issues with a
  two-week objection window before a result is marked verified.

## Roles

- **Maintainers** merge pull requests, run evaluations, publish releases, and hold the
  credentials for datasets and the leaderboard.
- **Contributors** propose sources, engines, code and documentation via pull requests.
- **Submitters** propose an engine for evaluation and sign the disclosure form.

## Decisions

Routine decisions are made by the maintainer handling the issue. Changes to the
methodology, licence policy, or this document require a pull request that stays open for
at least one week and is approved by all active maintainers.

## Releases

Dataset releases and results releases are versioned independently. Each release is a git
tag, has a changelog entry, and (for datasets) a Zenodo DOI. Nothing is removed from a
release retroactively; corrections are new releases with tombstones.
