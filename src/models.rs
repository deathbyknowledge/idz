use rusqlite::{Row, Result as RusqliteResult};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

/// Represents a chunk of text and its associated metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_id: String,
    pub content: String,
    pub metadata: Json,
}

impl<'stmt> TryFrom<&Row<'stmt>> for Chunk {
    type Error = rusqlite::Error;

    fn try_from(row: &Row<'stmt>) -> RusqliteResult<Self> {
        let chunk_id: String = row.get(0)?;
        let content: String = row.get(1)?;
        let metadata_str: String = row.get(2)?;

        let metadata: Json = serde_json::from_str(&metadata_str)
            .unwrap_or_else(|_| Json::Object(Default::default())); // Default to empty JSON object on error

        Ok(Chunk {
            chunk_id,
            content,
            metadata,
        })
    }
}

/// Represents a query vector, which can be of different underlying types.
#[derive(Debug, Clone)]
pub enum QueryVector<'a> {
    F32(&'a [f32]),
    // TODO: Add variants for I8, F16 etc.
    // I8(&'a [i8]),
}

/// Represents a search result, including the chunk and its distance to the query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub distance: f32,
}
