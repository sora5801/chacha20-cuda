# `docs/` — the per-push change log

This folder holds a **numbered, human-readable change document for every push**
to the repository. The rule for this project is simple:

> **Whenever something new is pushed to GitHub, add a new `NNNN-short-title.md`
> here describing what was added and why.**

These are not auto-generated commit messages. Each file is a short, didactic
write-up — what changed, the reasoning behind it, and anything a reader (often
future-me) should know. Because this repository is study material, the change
log doubles as a learning journal.

## Naming

```
docs/0001-initial-implementation.md
docs/0002-<short-title>.md
docs/0003-<short-title>.md
...
```

- Four-digit zero-padded sequence number, so the files sort in order.
- A short kebab-case title after the number.

## What each entry should contain

1. **Summary** — one or two sentences: what this push adds.
2. **Details** — the files touched and what each change does.
3. **Why** — the reasoning / trade-offs, not just the "what".
4. **Verification** — how it was tested (e.g. "all RFC 8439 vectors pass").

## Index

| # | Document | Summary |
|---|----------|---------|
| 0001 | [0001-initial-implementation.md](0001-initial-implementation.md) | Initial ChaCha20 CUDA implementation, tests, demo, and build files. |
