# coding-challenges-2025

This is a collection of coding challenges I have
found across the internet and decided to achieve in
2025.

## Table Of Contents

<!-- toc -->

- [Challenges](#challenges)
- [Contributing](#contributing)
  * [ASDF](#asdf)
  * [Pre-commit](#pre-commit)

<!-- tocstop -->

## Challenges

1. December, 2025 - `grpc` challenges.
   I would like to implement the same logic in both `golang` and `rust`.

1. December, 2025 - `volcano-sceduler` challenge.
   See the
   [volcano project](./machine-learning/volcano/).

1. December, 2025 - `ollama-kubernetes` challenges.
   See the
   [ollama project](./machine-learning/ollama/).

1. December, 2025 - `kuberay-mlflow` challenges.
   See the
   [KubeRay - MLFlow project](./machine-learning/kuberay-mlflow/).

1. December, 2025 - `custom-cni` challenges.
   See the
   [Custom CNI project](./kubernetes/custom-cni/).

## Contributing

### ASDF

This repository is versioned by
[`asdf`](https://asdf-vm.com/guide/getting-started.html).

You can install all `asdf`-versioned binaries via:

```bash
for plugin in $(awk '{ print $1 }' .tool-versions); do asdf plugin add "$plugin"; done
asdf install
asdf reshim
```

### Pre-commit

This repository uses
[pre-commit](https://pre-commit.com/)
to run a bunch of hooks before committing code.
This helps enforce things like code quality, formatting, etc.
before getting to the PR stage.

To install `pre-commmit` hooks:

```bash
pre-commit install
```

Or to run all `pre-commit` hooks:

```bash
pre-commit run --all-files
```
