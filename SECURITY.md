# Security policy

PIM Foundry is a research tool, not a sandbox for untrusted workloads.
Python backend and application plugins execute trusted code. Native simulator
paths and build scripts can run local programs. Replay bundles contain Python
engine snapshots and must be reviewed before execution. File hashes check
integrity; they do not authenticate the author or make code safe.

The supported public version is the latest tagged release. Report security
issues using GitHub's private vulnerability reporting for this repository when
available. Do not put credentials, private datasets, unreleased models or working
exploits against another person's service into public issues. Ordinary crashes
with a minimal synthetic reproducer can use the bug-report issue template.
