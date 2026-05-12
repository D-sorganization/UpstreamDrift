//! `notify`-based filesystem watcher for incremental re-indexing.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use anyhow::Result;
use notify::{Event, EventKind, RecursiveMode, Watcher};

use crate::indexer;
use crate::parsers;

/// Watch `repo_root` and re-index touched files. Blocks forever (Ctrl-C to stop).
pub fn watch(repo_root: &Path, debounce: Duration) -> Result<()> {
    let repo = repo_root
        .canonicalize()
        .unwrap_or_else(|_| repo_root.to_path_buf());
    let (tx, rx) = channel::<notify::Result<Event>>();
    let mut watcher = notify::recommended_watcher(move |res| {
        let _ = tx.send(res);
    })?;
    watcher.watch(&repo, RecursiveMode::Recursive)?;
    eprintln!("upstream-codemap: watching {}", repo.display());

    // Coalesce events within `debounce` window.
    let mut pending: HashSet<PathBuf> = HashSet::new();
    let mut last_event: Option<Instant> = None;

    loop {
        let timeout = match last_event {
            Some(t) => debounce
                .saturating_sub(t.elapsed())
                .max(Duration::from_millis(50)),
            None => Duration::from_secs(60),
        };
        match rx.recv_timeout(timeout) {
            Ok(Ok(ev)) => {
                if !matches!(
                    ev.kind,
                    EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
                ) {
                    continue;
                }
                for p in ev.paths {
                    if is_relevant(&p, &repo) {
                        pending.insert(p);
                    }
                }
                last_event = Some(Instant::now());
            }
            Ok(Err(_)) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if last_event.is_some() && !pending.is_empty() {
                    let start = Instant::now();
                    let mut n = 0u32;
                    for path in pending.drain() {
                        let rel = match path.strip_prefix(&repo) {
                            Ok(r) => r.to_string_lossy().replace('\\', "/"),
                            Err(_) => continue,
                        };
                        if let Err(e) = indexer::reindex_file(&repo, &rel) {
                            eprintln!("  reindex {rel} failed: {e}");
                        } else {
                            n += 1;
                        }
                    }
                    let ms = start.elapsed().as_millis();
                    eprintln!("  re-indexed {n} file(s) in {ms} ms");
                    last_event = None;
                }
            }
        }
    }
    Ok(())
}

fn is_relevant(path: &Path, repo: &Path) -> bool {
    if !path.starts_with(repo) {
        return false;
    }
    // Skip the index DB itself.
    if path
        .components()
        .any(|c| c.as_os_str() == ".codemap" || c.as_os_str() == ".git")
    {
        return false;
    }
    parsers::language_for(path).is_some()
}
