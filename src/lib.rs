// Re-used and new imports aligned with the new spec.
use std::path::Path;
use std::sync::{Arc, RwLock};

use hnsw_rs::prelude::*;
use rusqlite::backup;
use rusqlite::{params, Connection};
use serde_json::Value as Json;
use uuid::Uuid;

// --- Module Organization ---

/// Defines the primary error type for all library operations.
pub mod errors;
/// Defines the data models used in the library's public API.
pub mod models;

use crate::errors::DiskError;
use crate::models::{Chunk, QueryVector, SearchResult};

/// An enum to hold a type-erased HNSW index.
/// This allows the IdentityDisk to handle different vector types (f32, i8, etc.)
/// discovered at runtime from the model_signature.
pub enum SearchIndex { // Made public
    F32(Hnsw<'static, f32, DistCosine>),
    // TODO: Add variants for I8, F16 with appropriate distance metrics
    // I8(Hnsw<i8, SomeIntDistance>),
    None, // For disks opened without a supported index
}

// --- Constants ---

const SPEC_VERSION: &str = "1.0";

const CREATE_DB_SQL: &str = r#"
BEGIN;

CREATE TABLE manifest (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata TEXT -- Stored as JSON string
);
-- Index for faster chunk retrieval by ID
CREATE UNIQUE INDEX idx_chunks_chunk_id ON chunks(chunk_id);

CREATE TABLE indices (
    index_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL,
    index_type TEXT NOT NULL,
    model_signature TEXT NOT NULL,
    data BLOB NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE
);
-- Index for preventing duplicates and for fast loading
CREATE UNIQUE INDEX idx_indices_chunk_model ON indices (chunk_id, model_signature);
-- Index for faster loading of all indices for a given model
CREATE INDEX idx_indices_model_signature ON indices (model_signature);

INSERT INTO manifest (key, value) VALUES ('spec_version', ?1);

COMMIT;
"#;

/// The main interface for interacting with an Identity Disk (`.aim` file).
///
/// This struct holds a connection to the SQLite database and manages an
/// in-memory HNSW index for fast semantic search.
pub struct IdentityDisk {
    conn: Connection,
    index: Arc<RwLock<SearchIndex>>,
    // Maps the HNSW internal sequential ID to the database chunk_id (UUID)
    id_to_chunk_id: Arc<RwLock<Vec<String>>>,
    // The model signature this disk instance is actively managing
    model_signature: String,
}

impl IdentityDisk {
    /// Creates a new, empty Identity Disk at the specified path.
    ///
    /// If a file already exists at the path, it will be overwritten.
    ///
    /// # Arguments
    /// * `path` - The file path for the new disk.
    /// * `model_signature` - The model signature for the embeddings that will be stored.
    ///   e.g., "openai/text-embedding-3-small-1536_fp16"
    pub fn create<P: AsRef<Path>>(path: P, model_signature: &str) -> Result<Self, DiskError> {
        // Ensure we overwrite by deleting if it exists
        if path.as_ref().exists() {
            std::fs::remove_file(&path)?;
        }

        let conn = Connection::open(&path)?;
        conn.execute_batch(&CREATE_DB_SQL.replace("?1", &format!("'{}'", SPEC_VERSION)))?;

        let (index, _) = Self::load_index_from_db(&conn, model_signature)?;

        Ok(Self {
            conn,
            index: Arc::new(RwLock::new(index)),
            id_to_chunk_id: Arc::new(RwLock::new(Vec::new())),
            model_signature: model_signature.to_string(),
        })
    }

    /// Opens an existing Identity Disk.
    ///
    /// This will load all embeddings corresponding to the provided `model_signature`
    /// into an in-memory HNSW index for fast searching. If the signature is not
    /// found, it will open the disk with an empty search index.
    ///
    /// # Arguments
    /// * `path` - The file path of the disk to open.
    /// * `model_signature` - The specific model signature to load for searching.
    pub fn open<P: AsRef<Path>>(path: P, model_signature: &str) -> Result<Self, DiskError> {
        // TODO: Validate spec version from manifest table
        let conn = Connection::open(path)?;

        let (index, id_to_chunk_id) = Self::load_index_from_db(&conn, model_signature)?;
        Ok(Self {
            conn,
            index: Arc::new(RwLock::new(index)),
            id_to_chunk_id: Arc::new(RwLock::new(id_to_chunk_id)),
            model_signature: model_signature.to_string(),
        })
    }

    pub fn open_in_memory<P: AsRef<Path>>(
        path: P,
        model_signature: &str,
    ) -> Result<Self, DiskError> {
        let disk_conn = Connection::open(path)?;
        let mut mem_conn = Connection::open_in_memory()?;

        // Use the backup API to copy disk contents to memory
        {
            let backup = backup::Backup::new(&disk_conn, &mut mem_conn)?;
            backup.run_to_completion(5, std::time::Duration::from_millis(250), None)?;
        } // backup is dropped here, releasing the borrow

        let (index, id_to_chunk_id) = Self::load_index_from_db(&mem_conn, model_signature)?;

        Ok(Self {
            conn: mem_conn,
            index: Arc::new(RwLock::new(index)),
            id_to_chunk_id: Arc::new(RwLock::new(id_to_chunk_id)),
            model_signature: model_signature.to_string(),
        })
    }

    /// Adds a new chunk and its corresponding embedding to the disk.
    ///
    /// The operation is transactional. Both the chunk and its vector index
    /// are saved, or neither is.
    ///
    /// # Arguments
    /// * `content` - The text content of the chunk.
    /// * `embedding` - A slice representing the vector embedding.
    /// * `metadata` - Optional JSON metadata for the chunk.
    ///
    /// # Returns
    /// The unique `chunk_id` (UUID) of the newly added chunk.
    pub fn add_chunk(
        &mut self,
        content: &str,
        embedding: QueryVector,
        metadata: Option<Json>,
    ) -> Result<String, DiskError> {
        let chunk_id = Uuid::new_v4().to_string();
        let metadata_str = metadata.map_or("{}".to_string(), |j| j.to_string());

        let embedding_bytes: Vec<u8>;
        let vector_for_hnsw: &[f32]; // Temp, will be generic later

        // Match the input vector to serialize it correctly
        match embedding {
            QueryVector::F32(v) => {
                // Ensure the provided vector type matches the disk's index type
                if !self.model_signature.ends_with("_fp32") && !self.model_signature.contains('_') {
                    // default is fp32
                    return Err(DiskError::InvalidData(
                        "Mismatched vector type: expected fp32".into(),
                    ));
                }
                embedding_bytes = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                vector_for_hnsw = v;
            } // TODO: Add cases for I8, F16 etc.
        }

        // Use a transaction for atomicity
        let tx = self.conn.transaction()?;

        // 1. Insert chunk
        tx.execute(
            "INSERT INTO chunks (chunk_id, content, metadata) VALUES (?1, ?2, ?3)",
            params![&chunk_id, content, &metadata_str],
        )?;

        // 2. Insert index
        tx.execute(
            "INSERT INTO indices (chunk_id, index_type, model_signature, data) VALUES (?1, ?2, ?3, ?4)",
            params![&chunk_id, "vector_embedding", &self.model_signature, &embedding_bytes],
        )?;

        tx.commit()?;

        // Update in-memory HNSW index using enum dispatch
        {
            let mut id_map = self.id_to_chunk_id.write()?;
            let new_hnsw_id = id_map.len();
            let mut index = self.index.write()?;

            match &mut *index {
                SearchIndex::F32(ref mut hnsw) => {
                    hnsw.insert((&vector_for_hnsw, new_hnsw_id));
                }
                SearchIndex::None => {
                    // Cannot insert into a non-existent index.
                    return Err(DiskError::InvalidData("No supported index loaded.".into()));
                } // TODO: Handle other types
            }

            id_map.push(chunk_id.clone());
        }

        Ok(chunk_id)
    }

    /// Retrieves all chunks from the disk, without their vector embeddings.
    pub fn get_chunks(&self) -> Result<Vec<Chunk>, DiskError> {
        let mut stmt = self
            .conn
            .prepare("SELECT chunk_id, content, metadata FROM chunks")?;
        let chunk_iter = stmt.query_map([], |row| Chunk::try_from(row))?;

        let mut chunks = Vec::new();
        for chunk in chunk_iter {
            chunks.push(chunk?);
        }
        Ok(chunks)
    }

    /// Performs a semantic search for the `top_k` most similar chunks.
    ///
    /// # Arguments
    /// * `query_vector` - The embedding of the query.
    /// * `top_k` - The number of results to return.
    ///
    /// # Returns
    /// A vector of `SearchResult` structs, sorted by similarity.
    pub fn search(
        &self,
        query_vector: QueryVector,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, DiskError> {
        let index = self.index.read()?;
        let neighbors = match (&*index, query_vector) {
            (SearchIndex::F32(hnsw), QueryVector::F32(q)) => {
                hnsw.search(q, top_k, 100)
            },
            // Mismatched types
            // (SearchIndex::F32(_), _) => return Err(DiskError::InvalidData("Search query type does not match index type (f32).".into())),
            (SearchIndex::None, _) => return Ok(Vec::new()), // No index, no results
        };

        let id_map = self.id_to_chunk_id.read()?;
        let mut results: Vec<SearchResult> = Vec::with_capacity(neighbors.len());
        for neighbor in neighbors {
            let chunk_id = &id_map[neighbor.d_id];

            let mut stmt = self
                .conn
                .prepare("SELECT chunk_id, content, metadata FROM chunks WHERE chunk_id = ?1")?;
            let chunk = stmt.query_row(params![chunk_id], |row| Chunk::try_from(row))?;

            results.push(SearchResult {
                chunk,
                distance: neighbor.distance,
            });
        }

        Ok(results)
    }

    /// Updates the metadata of an existing chunk.
    ///
    /// Note: This does not allow changing the `content` of a chunk, as that
    /// would invalidate its embedding.
    pub fn update_chunk_metadata(
        &mut self,
        chunk_id: &str,
        new_metadata: Json,
    ) -> Result<(), DiskError> {
        let metadata_str = new_metadata.to_string();
        let rows_affected = self.conn.execute(
            "UPDATE chunks SET metadata = ?1 WHERE chunk_id = ?2",
            params![metadata_str, chunk_id],
        )?;

        if rows_affected == 0 {
            Err(DiskError::NotFound(chunk_id.to_string()))
        } else {
            Ok(())
        }
    }

    /// Helper to load the index, now with type dispatching.
    fn load_index_from_db(
        conn: &Connection,
        model_signature: &str,
    ) -> Result<(SearchIndex, Vec<String>), DiskError> {
        // --- This block is now more memory efficient ---
        // It collects all blobs first, then processes them, avoiding per-row vector allocation.
        let mut stmt = conn.prepare(
            "SELECT chunk_id, data FROM indices WHERE model_signature = ?1 ORDER BY chunk_id",
        )?;
        let mut rows = stmt.query(params![model_signature])?;

        let mut id_map = Vec::new();
        let mut data_blobs = Vec::new();
        while let Some(row) = rows.next()? {
            id_map.push(row.get(0)?);
            data_blobs.push(row.get::<_, Vec<u8>>(1)?);
        }
        // --- End memory efficient block ---

        if id_map.is_empty() {
            return Ok((SearchIndex::None, Vec::new()));
        }

        // Dispatch based on signature
        if model_signature.ends_with("_fp32") || !model_signature.contains('_') {
            // Default to f32
            let num_items = id_map.len();
            let hnsw: Hnsw<'static, f32, DistCosine> =
                Hnsw::new(16, num_items.max(1), 16, 200, DistCosine {});

            // This loop is still necessary to build the index.
            for (i, blob) in data_blobs.iter().enumerate() {
                let vector: Vec<f32> = blob
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                // We use insert_slice later if we can collect all vectors into one slice
                hnsw.insert((&vector, i));
            }
            Ok((SearchIndex::F32(hnsw), id_map))
        }
        // TODO: Add `else if` blocks for `_fp16`, `_int8`, etc.
        // else if model_signature.ends_with("_int8") { ... }
        else {
            // Signature is present but not supported by this library version
            eprintln!("Warning: Unsupported vector type for model signature '{}'. Search will be disabled.", model_signature);
            Ok((SearchIndex::None, id_map))
        }
    }

    /// Retrieves the specification version of the disk.
    pub fn get_spec_version(&self) -> Result<String, DiskError> {
        let version = self.conn.query_row(
            "SELECT value FROM manifest WHERE key = 'spec_version'",
            [],
            |row| row.get(0),
        )?;
        Ok(version)
    }

    /// Returns the type of the currently loaded search index.
    pub fn get_index_type_description(&self) -> Result<String, DiskError> {
        let index_guard = self.index.read()?;
        Ok(match *index_guard {
            SearchIndex::F32(_) => "F32 (Cosine Distance)".to_string(),
            SearchIndex::None => "None (No index loaded or supported for current model signature)".to_string(),
            // Add other types as they are implemented
        })
    }
}
