use std::fs::{File, OpenOptions};
use std::io::{Cursor, Read, Write};
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_array::{Int64Array, LargeBinaryArray, LargeStringArray};
use arrow_schema::{DataType, Field, Schema, ArrowError};
use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};
use chrono::Utc;
use hnsw_rs::prelude::*;
use parquet::arrow::{arrow_writer::ArrowWriter, arrow_reader::ParquetRecordBatchReader};
use parquet::file::properties::WriterProperties;
use parquet::basic::Compression;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use thiserror::Error;
use zip::write::SimpleFileOptions;
use zip::{ZipArchive, ZipWriter};


// -----------------------------------------------------------------------------
// Constants — internal paths inside the .idz ZIP
// -----------------------------------------------------------------------------
const MANIFEST_PATH: &str = "manifest.json";
const PARQUET_PATH: &str = "text.parquet"; // Arrow Parquet, ZSTD‑compressed
const EMBED_PATH: &str = "embeds.f32";    // little‑endian row‑major f32
const HNSW_PATH: &str = "index.hnsw";      // binary dump produced by hnsw_rs

// -----------------------------------------------------------------------------
// Errors
// -----------------------------------------------------------------------------
#[derive(Debug, Error)]
pub enum DiskError {
    #[error("zip error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serde json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("arrow error: {0}")]
    Arrow(#[from] ArrowError),
    #[error("invalid embedding dimension")] InvalidDim,
}

// -----------------------------------------------------------------------------
// Manifest — tiny JSON descriptor kept uncompressed at the front
// -----------------------------------------------------------------------------
#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u8,
    pub created: String,
    pub embedding: EmbeddingInfo,
    pub chunks: ChunkInfo,
    pub index: IndexInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingInfo {
    pub model: String,
    pub dim: usize,
    pub dtype: String,   // "float32" | "int8"
    pub quantised: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChunkInfo {
    pub count: usize,
    pub average_chars: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexInfo {
    pub kind: String,         // "HNSW"
    pub ef_construct: usize,
    pub m: usize,
}

// -----------------------------------------------------------------------------
// Builder
// -----------------------------------------------------------------------------
pub struct IdentityDiskBuilder {
    dim: usize,
    model: String,
    texts: Vec<String>,
    metas: Vec<Json>,
    embeds: Vec<f32>,
    // HNSW parameters
    m: usize,
    ef_construct: usize,
}

impl IdentityDiskBuilder {
    pub fn new(dim: usize, model: &str) -> Self {
        Self { dim, model: model.into(), texts: vec![], metas: vec![], embeds: vec![], m: 16, ef_construct: 200 }
    }

    pub fn with_hnsw_params(mut self, m: usize, ef: usize) -> Self {
        self.m = m;
        self.ef_construct = ef;
        self
    }

    /// Add one (text, meta, embedding) triple
    pub fn push(&mut self, text: &str, meta: Json, embedding: &[f32]) -> Result<(), DiskError> {
        if embedding.len() != self.dim {
            return Err(DiskError::InvalidDim);
        }
        self.texts.push(text.to_owned());
        self.metas.push(meta);
        self.embeds.extend_from_slice(embedding);
        Ok(())
    }

    /// Finish and write .idz file
    pub fn write<P: AsRef<Path>>(self, path: P) -> Result<(), DiskError> {
        // --------------------------------------------------------------
        // 1. Build Parquet in‑memory
        // --------------------------------------------------------------
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("text", DataType::LargeUtf8, false),
            Field::new("meta", DataType::LargeBinary, false),
        ]));

        let mut parquet_buf = Vec::new();
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .build();
        let mut writer = ArrowWriter::try_new(&mut parquet_buf, schema.clone(), Some(props))?;

        let batch = RecordBatch::try_new(schema, vec![
            arc_array(Int64Array::from_iter_values(0i64..self.texts.len() as i64)),
            arc_array(LargeStringArray::from(self.texts.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
            arc_array(LargeBinaryArray::from(
                self.metas.iter()
                    .map(|v| serde_json::to_vec(v).unwrap())
                    .collect::<Vec<Vec<u8>>>()
                    .iter()
                    .map(|v| v.as_slice())
                    .collect::<Vec<&[u8]>>()
            )),
        ])?;
        writer.write(&batch)?;
        writer.close()?;

        // --------------------------------------------------------------
        // 2. Build HNSW
        // --------------------------------------------------------------
        let hnsw: Hnsw<'static, f32, DistL2> = Hnsw::new(
            self.m,
            self.texts.len(),
            16, // max_layer
            self.ef_construct,
            DistL2 {},
        );

        for (i, chunk) in self.embeds.chunks(self.dim).enumerate() {
            hnsw.insert((chunk, i));
        }

        // For now, let's use a simpler approach to save the HNSW
        let hnsw_bytes = vec![]; // placeholder - we'll build this later

        // --------------------------------------------------------------
        // 3. Create ZIP
        // --------------------------------------------------------------
        let file = File::create(path)?;
        let mut zip = ZipWriter::new(file);
        let store = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        let deflate = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

        // Manifest
        let avg_chars = self.texts.iter().map(|t| t.len()).sum::<usize>() / self.texts.len().max(1);
        let manifest = Manifest {
            version: 1,
            created: Utc::now().to_rfc3339(),
            embedding: EmbeddingInfo { model: self.model, dim: self.dim, dtype: "float32".into(), quantised: false },
            chunks: ChunkInfo { count: self.texts.len(), average_chars: avg_chars },
            index: IndexInfo { kind: "HNSW".into(), ef_construct: self.ef_construct, m: self.m },
        };
        zip.start_file(MANIFEST_PATH, store)?;
        zip.write_all(serde_json::to_string_pretty(&manifest)?.as_bytes())?;

        // Parquet
        zip.start_file(PARQUET_PATH, deflate)?;
        zip.write_all(&parquet_buf)?;

        // Embeddings raw
        zip.start_file(EMBED_PATH, store)?;
        let mut raw = Vec::with_capacity(self.embeds.len() * 4);
        let mut cur = Cursor::new(&mut raw);
        for &v in &self.embeds { cur.write_f32::<LittleEndian>(v)?; }
        zip.write_all(&raw)?;

        // HNSW dump
        zip.start_file(HNSW_PATH, deflate)?;
        zip.write_all(&hnsw_bytes)?;

        zip.finish()?;
        Ok(())
    }
}

// Helper wrapper to avoid long Arc::new(...) lines
fn arc_array<T: arrow_array::Array + 'static>(a: T) -> ArrayRef {
    std::sync::Arc::new(a)
}

// -----------------------------------------------------------------------------
// Reader / query interface
// -----------------------------------------------------------------------------
pub struct IdentityDisk {
    pub manifest: Manifest,
    pub texts: Vec<String>,
    pub metas: Vec<Json>,
    pub _embeds: Vec<f32>,
    pub hnsw: Hnsw<'static, f32, DistL2>,
}

impl IdentityDisk {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, DiskError> {
        let file = OpenOptions::new().read(true).open(&path)?;
        let mut zip = ZipArchive::new(file)?;

        // Manifest
        let mut mf_str = String::new();
        zip.by_name(MANIFEST_PATH)?.read_to_string(&mut mf_str)?;
        let manifest: Manifest = serde_json::from_str(&mf_str)?;

        // Parquet -> Arrow
        let mut parquet_bytes = Vec::new();
        zip.by_name(PARQUET_PATH)?.read_to_end(&mut parquet_bytes)?;
        
        let reader = ParquetRecordBatchReader::try_new(bytes::Bytes::from(parquet_bytes), 1024)?;
        let mut texts = Vec::new();
        let mut metas = Vec::new();
        for batch in reader {
            let batch = batch?;
            let ids = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let txts = batch.column(1).as_any().downcast_ref::<LargeStringArray>().unwrap();
            let mbs = batch.column(2).as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            for row in 0..batch.num_rows() {
                let _id = ids.value(row) as usize; // guaranteed order
                texts.push(txts.value(row).to_string());
                metas.push(serde_json::from_slice(mbs.value(row)).unwrap_or(Json::Null));
            }
        }

        // Read embeddings from ZIP
        let mut embeds_data = Vec::new();
        zip.by_name(EMBED_PATH)?.read_to_end(&mut embeds_data)?;
        
        // Convert bytes to f32 values
        let mut embeds = Vec::with_capacity(embeds_data.len() / 4);
        let mut cursor = Cursor::new(&embeds_data);
        while cursor.position() < embeds_data.len() as u64 {
            embeds.push(cursor.read_f32::<LittleEndian>()?);
        }

        // Build HNSW index from scratch (since serialization is complex)
        let hnsw: Hnsw<'static, f32, DistL2> = Hnsw::new(
            manifest.index.m,
            texts.len(),
            16, // max_layer
            manifest.index.ef_construct,
            DistL2 {},
        );

        for (i, chunk) in embeds.chunks(manifest.embedding.dim).enumerate() {
            hnsw.insert((chunk, i));
        }

        Ok(Self { manifest, texts, metas, _embeds: embeds, hnsw })
    }

    /// Query by embedding vector, returns (id, score) pairs
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.hnsw.search(query, k, self.manifest.index.ef_construct).iter()
            .map(|neig| (neig.d_id as usize, neig.distance)).collect()
    }

    pub fn chunk_text(&self, id: usize) -> &str { &self.texts[id] }
    pub fn chunk_meta(&self, id: usize) -> &Json { &self.metas[id] }
}
