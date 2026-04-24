# .gaai/ — GAAI Framework (v2.6.3)

## Directory Structure

```
.gaai/
├── core/          ← Framework (auto-synced to OSS via post-commit hook)
│   └── README.md  ← This file
└── project/       ← Project-specific data (memory, backlog, artefacts, custom skills)
```

- `core/` changes are **automatically contributed to OSS** on every commit (via post-commit hook → PR → auto-merge)
- `project/` is **local only** — never synced to OSS

---

## Framework Sync (Automatic)

When you commit changes to `.gaai/core/`, a post-commit hook automatically:
