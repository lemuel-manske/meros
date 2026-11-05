#!/usr/bin/env bash
set -euo pipefail

name="${1:-}"
message="${2:-}"
date="${3:-$(date +%F)}"

if [[ ! "$name" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
    echo "Usage: make checkpoint NAME=topic [MESSAGE='Research outcome'] [DATE=YYYY-MM-DD]" >&2
    exit 2
fi

if [[ ! "$date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "DATE must be YYYY-MM-DD" >&2
    exit 2
fi

branch="$(git symbolic-ref --quiet --short HEAD)" || {
    echo "Checkpoints require an experimental branch (not detached HEAD)." >&2
    exit 1
}

if [[ "$branch" == "main" || "$branch" == "master" ]]; then
    echo "Cannot checkpoint the maintained branch: $branch" >&2
    exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    echo "Commit or discard tracked/untracked changes before creating a checkpoint." >&2
    exit 1
fi

tag="research/$name/$date"

if git show-ref --verify --quiet "refs/tags/$tag"; then
    echo "Tag already exists locally: $tag" >&2
    exit 1
fi

remote_tag="$(git ls-remote --tags origin "refs/tags/$tag")"
if [[ -n "$remote_tag" ]]; then
    echo "Tag already exists on origin: $tag" >&2
    exit 1
fi

# Only upload DVC data when the experiment changed DVC tracking metadata.
# An unchanged pointer may already reference data present remotely.
if git show-ref --verify --quiet refs/remotes/origin/main; then
    changed_dvc="$(
        git diff --name-only "$(git merge-base origin/main HEAD)" HEAD --             '*.dvc' 'dvc.lock'
    )"
else
    echo "Fetch origin/main before checkpointing so DVC changes can be checked." >&2
    exit 1
fi

if [[ -n "$changed_dvc" ]]; then
    echo "DVC tracking changes detected; uploading objects before publishing the tag."
    if ! python -m dvc --version >/dev/null 2>&1; then
        echo "Install DVC and configure its remote before checkpointing." >&2
        exit 1
    fi

    python -m dvc push
fi

echo "Publishing branch $branch..."
git push origin "HEAD:refs/heads/$branch"

if [[ -z "$message" ]]; then
    message="Research checkpoint: $name"
fi

git tag -a "$tag" -m "$message"
git push origin "refs/tags/$tag:refs/tags/$tag"

expected="$(git rev-parse HEAD)"
actual="$(git ls-remote --tags origin "refs/tags/$tag^{}" | awk '{print $1}')"

if [[ "$actual" != "$expected" ]]; then
    echo "Could not verify remote annotated tag $tag points to $expected" >&2
    exit 1
fi

echo "Checkpoint published and verified: $tag ($expected)"
echo "The branch may be deleted only after any required artifact recovery is verified."
