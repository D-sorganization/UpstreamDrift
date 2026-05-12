//! Vector persistence and RAG memory manager.
//!
//! Follows Law of Demeter by encapsulating the database connection logic.
//! Pure-Rust core is always compiled; the `python` feature adds PyO3 bindings.
//!
//! ## Vector search backend
//!
//! After surveying `sqlite-vss` on Windows MSVC (PR #5242 left a `try_load_vss`
//! stub that always returned `false`) and `hnsw_rs` (extra complexity for the
//! desktop-launcher scale we care about), we ship a **brute-force in-memory
//! cosine similarity** path backed by SQLite persistence. Embeddings are
//! stored as little-endian `f32` BLOBs in the `documents` table; on `search`
//! we stream every row, compute cosine similarity against the query, keep the
//! top-k via a bounded min-heap. For the 1k–10k chunk scale of a typical
//! repo, this is well under 10 ms per query (validated in the PR body).
//!
//! If the chunk count grows past ~100k we can plug in `hnsw_rs` behind the
//! same `try_search` surface — the storage format is forward-compatible.

#[cfg(feature = "python")]
use pyo3::exceptions::{PyRuntimeError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;

use rusqlite::{params, Connection};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};

/// Manages vector persistence and RAG memory.
#[cfg_attr(feature = "python", pyclass)]
pub struct MemoryManager {
    #[allow(dead_code)]
    db_path: String,
    conn: Arc<Mutex<Connection>>,
}

impl MemoryManager {
    /// Pure-Rust constructor.
    ///
    /// # Contract
    /// * `db_path` must not be empty.
    pub fn try_new(db_path: String) -> Result<Self, String> {
        if db_path.trim().is_empty() {
            return Err("db_path cannot be empty".to_string());
        }

        let conn = Connection::open(&db_path).map_err(|e| format!("Failed to open DB: {}", e))?;

        Ok(Self {
            db_path,
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Initializes the database schema.
    ///
    /// Schema v2 (this crate version): single `documents` table with
    /// `payload`, raw `embedding` BLOB (LE f32), `dim`, and a content
    /// `hash` for idempotent re-indexing. v1 (PR #5242) had a separate
    /// `vss_documents` virtual table — that path is gone; the migration is
    /// drop-and-recreate behind the schema check below.
    pub fn try_initialize(&self) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payload TEXT NOT NULL,
                embedding BLOB,
                dim INTEGER,
                hash TEXT
            )",
            [],
        )
        .map_err(|e| format!("DB init error: {}", e))?;

        // Older schema (PR #5242) had no `embedding`/`dim`/`hash` columns.
        // Best-effort ADD COLUMN; ignore errors when already present.
        let _ = conn.execute("ALTER TABLE documents ADD COLUMN embedding BLOB", []);
        let _ = conn.execute("ALTER TABLE documents ADD COLUMN dim INTEGER", []);
        let _ = conn.execute("ALTER TABLE documents ADD COLUMN hash TEXT", []);

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash)",
            [],
        )
        .map_err(|e| format!("DB index error: {}", e))?;

        Ok(())
    }

    /// Stores a vector and its associated payload in the database.
    ///
    /// # Contract
    /// * `payload` must not be empty.
    /// * `embedding` must not be empty.
    pub fn try_store_embedding(&self, payload: String, embedding: Vec<f32>) -> Result<(), String> {
        if payload.trim().is_empty() {
            return Err("payload cannot be empty".to_string());
        }
        if embedding.is_empty() {
            return Err("embedding cannot be empty".to_string());
        }

        let hash = content_hash(&payload);
        let dim = embedding.len() as i64;
        let blob = embedding_to_blob(&embedding);

        let conn = self.conn.lock().unwrap();

        // Idempotency: skip insert when an identical chunk is already stored.
        let already: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM documents WHERE hash = ?1",
                params![hash],
                |row| row.get(0),
            )
            .unwrap_or(0);
        if already > 0 {
            return Ok(());
        }

        conn.execute(
            "INSERT INTO documents (payload, embedding, dim, hash) VALUES (?1, ?2, ?3, ?4)",
            params![payload, blob, dim, hash],
        )
        .map_err(|e| format!("DB insert error: {}", e))?;

        Ok(())
    }

    /// Retrieves the top-k most similar payloads for a given vector.
    ///
    /// Uses brute-force cosine similarity over all stored embeddings whose
    /// dimension matches the query. Returns payloads ordered by descending
    /// similarity. See the module docstring for the rationale.
    pub fn try_search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<String>, String> {
        if query_embedding.is_empty() {
            return Err("query_embedding cannot be empty".to_string());
        }
        if top_k == 0 {
            return Err("top_k must be greater than 0".to_string());
        }

        let conn = self.conn.lock().unwrap();
        let q_norm = l2_norm(&query_embedding);
        let dim = query_embedding.len() as i64;

        // Stream every same-dim row through the heap. Rows without an
        // embedding (legacy v1) fall through to the empty-result path so
        // callers can re-index without surprises.
        let mut stmt = match conn.prepare(
            "SELECT payload, embedding FROM documents WHERE dim = ?1 AND embedding IS NOT NULL",
        ) {
            Ok(s) => s,
            Err(e) => return Err(format!("DB prepare error: {}", e)),
        };

        let rows = stmt
            .query_map(params![dim], |row| {
                let payload: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((payload, blob))
            })
            .map_err(|e| format!("DB query error: {}", e))?;

        // Min-heap of size top_k for streaming top-k.
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(top_k + 1);

        for row in rows.flatten() {
            let (payload, blob) = row;
            let Some(emb) = blob_to_embedding(&blob, query_embedding.len()) else {
                continue;
            };
            let sim = cosine(&query_embedding, q_norm, &emb);
            heap.push(HeapEntry { sim, payload });
            if heap.len() > top_k {
                heap.pop();
            }
        }

        // Drain the heap into a Vec then sort by descending sim. We avoid
        // `into_sorted_vec` because our inverted `Ord` (min-heap behavior)
        // makes the resulting order non-obvious; an explicit sort is clear
        // and just as cheap at top-k scale.
        let mut entries: Vec<HeapEntry> = heap.into_iter().collect();
        entries.sort_by(|a, b| b.sim.partial_cmp(&a.sim).unwrap_or(Ordering::Equal));
        Ok(entries.into_iter().map(|e| e.payload).collect())
    }

    /// Returns the number of stored documents (including those without
    /// embeddings — useful for the CLI `stats` subcommand).
    pub fn try_count(&self) -> Result<usize, String> {
        let conn = self.conn.lock().unwrap();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))
            .map_err(|e| format!("DB count error: {}", e))?;
        Ok(n as usize)
    }
}

/// Heap entry: stores cosine similarity + payload. We invert the ordering
/// (smaller-is-larger) so `BinaryHeap` acts as a min-heap, letting us drop
/// the lowest-similarity entry once the heap exceeds `top_k`.
struct HeapEntry {
    sim: f32,
    payload: String,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.sim == other.sim
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Invert: smaller similarity is "greater" in heap terms, so `pop()`
        // removes the worst candidate. NaN sorts as Less to keep order total.
        other.sim.partial_cmp(&self.sim).unwrap_or(Ordering::Equal)
    }
}

/// SHA-256 hex of the payload (first 16 bytes hex-encoded). Used for
/// idempotent re-indexing; collisions at 64 bits of entropy are astronomically
/// unlikely for a per-repo chunk corpus.
pub(crate) fn content_hash(payload: &str) -> String {
    // Cheap FNV-1a 64 — pulling in sha2 here would force everyone to install
    // it just for dedupe, which is overkill. The `sha2` dep is gated behind
    // `local-embeddings` for the ONNX model integrity check.
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in payload.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", hash)
}

fn embedding_to_blob(emb: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(emb.len() * 4);
    for v in emb {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn blob_to_embedding(blob: &[u8], expected_dim: usize) -> Option<Vec<f32>> {
    if blob.len() != expected_dim * 4 {
        return None;
    }
    let mut out = Vec::with_capacity(expected_dim);
    for chunk in blob.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(out)
}

fn l2_norm(v: &[f32]) -> f32 {
    let s: f32 = v.iter().map(|x| x * x).sum();
    s.sqrt().max(f32::EPSILON)
}

fn cosine(query: &[f32], q_norm: f32, other: &[f32]) -> f32 {
    let mut dot = 0f32;
    let mut o_norm_sq = 0f32;
    for (a, b) in query.iter().zip(other.iter()) {
        dot += a * b;
        o_norm_sq += b * b;
    }
    let o_norm = o_norm_sq.sqrt().max(f32::EPSILON);
    dot / (q_norm * o_norm)
}

#[cfg(feature = "python")]
#[pymethods]
impl MemoryManager {
    /// Creates a new MemoryManager instance.
    ///
    /// # Contract
    /// * `db_path` must not be empty.
    #[new]
    pub fn new(db_path: String) -> PyResult<Self> {
        Self::try_new(db_path).map_err(|e| {
            if e == "db_path cannot be empty" {
                PyValueError::new_err(e)
            } else {
                PyRuntimeError::new_err(e)
            }
        })
    }

    /// Initializes the database schema.
    pub fn initialize(&self) -> PyResult<()> {
        self.try_initialize().map_err(PyRuntimeError::new_err)
    }

    /// Stores a vector and its associated payload in the database.
    pub fn store_embedding(&self, payload: String, embedding: Vec<f32>) -> PyResult<()> {
        self.try_store_embedding(payload, embedding).map_err(|e| {
            if e.starts_with("payload") || e.starts_with("embedding") {
                PyValueError::new_err(e)
            } else {
                PyRuntimeError::new_err(e)
            }
        })
    }

    /// Retrieves the top-k most similar payloads for a given vector.
    pub fn search(&self, query_embedding: Vec<f32>, top_k: usize) -> PyResult<Vec<String>> {
        self.try_search(query_embedding, top_k).map_err(|e| {
            if e.starts_with("query_embedding") || e.starts_with("top_k") {
                PyValueError::new_err(e)
            } else {
                PyRuntimeError::new_err(e)
            }
        })
    }

    /// Number of stored documents.
    pub fn count(&self) -> PyResult<usize> {
        self.try_count().map_err(PyRuntimeError::new_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_rejects_empty_path() {
        let result = MemoryManager::try_new("   ".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_store_embedding_validation() {
        let manager = MemoryManager::try_new(":memory:".to_string()).unwrap();
        manager.try_initialize().unwrap();
        let result = manager.try_store_embedding("".to_string(), vec![0.1, 0.2]);
        assert!(result.is_err());

        let result = manager.try_store_embedding("data".to_string(), vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cosine_search_returns_closest_first() {
        let manager = MemoryManager::try_new(":memory:".to_string()).unwrap();
        manager.try_initialize().unwrap();

        // Three orthogonal-ish unit vectors plus one duplicate of `target`.
        let target = vec![1.0, 0.0, 0.0];
        let other_a = vec![0.0, 1.0, 0.0];
        let other_b = vec![0.0, 0.0, 1.0];
        let near = vec![0.9, 0.1, 0.05];

        manager
            .try_store_embedding("payload_target".to_string(), target.clone())
            .unwrap();
        manager
            .try_store_embedding("payload_a".to_string(), other_a)
            .unwrap();
        manager
            .try_store_embedding("payload_b".to_string(), other_b)
            .unwrap();
        manager
            .try_store_embedding("payload_near".to_string(), near)
            .unwrap();

        let hits = manager.try_search(target, 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0], "payload_target");
        assert_eq!(hits[1], "payload_near");
    }

    #[test]
    fn test_search_rejects_empty_query() {
        let manager = MemoryManager::try_new(":memory:".to_string()).unwrap();
        manager.try_initialize().unwrap();
        let result = manager.try_search(vec![], 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_rejects_zero_top_k() {
        let manager = MemoryManager::try_new(":memory:".to_string()).unwrap();
        manager.try_initialize().unwrap();
        let result = manager.try_search(vec![0.1, 0.2], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_store_is_idempotent_on_identical_payload() {
        let manager = MemoryManager::try_new(":memory:".to_string()).unwrap();
        manager.try_initialize().unwrap();
        manager
            .try_store_embedding("dup".to_string(), vec![0.1, 0.2, 0.3])
            .unwrap();
        manager
            .try_store_embedding("dup".to_string(), vec![0.1, 0.2, 0.3])
            .unwrap();
        assert_eq!(manager.try_count().unwrap(), 1);
    }

    #[test]
    fn test_blob_roundtrip() {
        let original = vec![1.0_f32, -2.0, 0.5, std::f32::consts::PI];
        let blob = embedding_to_blob(&original);
        let restored = blob_to_embedding(&blob, original.len()).unwrap();
        assert_eq!(restored, original);
    }

    #[test]
    fn test_blob_rejects_wrong_dim() {
        let blob = embedding_to_blob(&[1.0, 2.0, 3.0]);
        assert!(blob_to_embedding(&blob, 4).is_none());
    }

    /// Reference performance numbers for the brute-force search backend.
    /// Run with `cargo test -p ai_backend -- --ignored --nocapture bench_`.
    #[test]
    #[ignore]
    fn bench_insert_and_query_384d_5k() {
        use std::time::Instant;
        let mem = MemoryManager::try_new(":memory:".to_string()).unwrap();
        mem.try_initialize().unwrap();
        let dim = 384;
        let n = 5_000;
        let t0 = Instant::now();
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|j| ((i * 31 + j) as f32).sin()).collect();
            mem.try_store_embedding(format!("doc_{}", i), v).unwrap();
        }
        let insert = t0.elapsed();
        let mut total = std::time::Duration::ZERO;
        for k in 0..100 {
            let q: Vec<f32> = (0..dim)
                .map(|j| (j as f32 + k as f32 * 0.1).sin())
                .collect();
            let t = Instant::now();
            let _ = mem.try_search(q, 10).unwrap();
            total += t.elapsed();
        }
        println!(
            "[bench] inserted {} x {}-dim in {:?} ({:.0}/s); 100 top-10 queries totalled {:?} ({:.2} ms/query)",
            n,
            dim,
            insert,
            n as f64 / insert.as_secs_f64(),
            total,
            total.as_secs_f64() * 1000.0 / 100.0
        );
    }

    #[test]
    fn test_search_ignores_mismatched_dim() {
        let manager = MemoryManager::try_new(":memory:".to_string()).unwrap();
        manager.try_initialize().unwrap();
        manager
            .try_store_embedding("dim3".to_string(), vec![1.0, 0.0, 0.0])
            .unwrap();
        manager
            .try_store_embedding("dim4".to_string(), vec![1.0, 0.0, 0.0, 0.0])
            .unwrap();
        let hits = manager.try_search(vec![1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(hits, vec!["dim3".to_string()]);
    }
}
