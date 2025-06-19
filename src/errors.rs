use thiserror::Error;
use rusqlite;
use serde_json;

#[derive(Debug, Error)]
pub enum DiskError {
    #[error("SQLite error: {0}")]
    Rusqlite(#[from] rusqlite::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization/deserialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Lock acquisition error (RwLock read): {0}")]
    RwLockRead(String), // Simplified from PoisonError

    #[error("Lock acquisition error (RwLock write): {0}")]
    RwLockWrite(String), // Simplified from PoisonError

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Chunk or resource not found: {0}")]
    NotFound(String),

    #[error("HNSW_RS error: {0}")]
    Hnsw(String), // hnsw_rs errors are often strings or require specific handling
}

// Implement From for RwLock PoisonErrors if you need to be more specific
impl<T> From<std::sync::PoisonError<T>> for DiskError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        DiskError::RwLockRead(format!("Failed to acquire lock: {}", err))
    }
}
