use async_trait::async_trait;
use rust_mcp_sdk::{
    error::SdkResult,
    macros::{self, JsonSchema},
    mcp_server::{server_runtime, McpServerOptions, ServerHandler},
    schema::{
        schema_utils::CallToolError, CallToolRequestParams, CallToolResult,
        Implementation, ListResourcesResult, ListResourcesResultCacheScope,
        ListToolsResult, ListToolsResultCacheScope, PaginatedRequestParams,
        ReadResourceRequestParams, ReadResourceResult, ReadResourceResultCacheScope,
        Resource, RpcError, ServerCapabilities, ServerCapabilitiesResources,
        ServerCapabilitiesTools, ServerResult, TextContent, TextResourceContents,
    },
    McpServer, RequestContext, ServerDetails, StdioTransport, ToMcpServerHandler,
    TransportOptions,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use tract_onnx::prelude::*;

const EMBED_DIM: usize = 384;

// --- Keyword index built at startup ---

struct Chunk {
    topic: String,
    stem: String,
    category: String,
    body_lower: String,
    full_text: String,
}

struct Index {
    chunks: Vec<Chunk>,
    chunks_dir: PathBuf,
}

fn parse_frontmatter(text: &str) -> (HashMap<&str, &str>, &str) {
    let mut fm = HashMap::new();
    let body = if text.starts_with("---") {
        if let Some(end) = text[3..].find("\n---") {
            let header = &text[3..3 + end];
            for line in header.lines() {
                if let Some((k, v)) = line.split_once(':') {
                    fm.insert(k.trim(), v.trim());
                }
            }
            text[3 + end + 4..].trim_start()
        } else {
            text
        }
    } else {
        text
    };
    (fm, body)
}

fn build_index(chunks_dir: &Path) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let Ok(entries) = std::fs::read_dir(chunks_dir) else {
        return chunks;
    };
    for cat_entry in entries.flatten() {
        if !cat_entry.path().is_dir() {
            continue;
        }
        let category = cat_entry.file_name().to_string_lossy().to_string();
        let Ok(files) = std::fs::read_dir(cat_entry.path()) else {
            continue;
        };
        for file_entry in files.flatten() {
            let path = file_entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let Ok(full_text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let (fm, body) = parse_frontmatter(&full_text);
            let topic = fm.get("topic").unwrap_or(&"").to_string();
            let stem = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            chunks.push(Chunk {
                topic,
                stem,
                category: category.clone(),
                body_lower: body.to_lowercase(),
                full_text,
            });
        }
    }
    chunks
}

// --- Semantic index (vectors + ONNX model) ---

#[derive(Deserialize)]
struct VectorMeta {
    path: String,
    topic: String,
    category: String,
}

struct SemanticIndex {
    vectors: Vec<[f32; EMBED_DIM]>,
    metadata: Vec<VectorMeta>,
    model: TypedRunnableModel<TypedModel>,
    tokenizer: tokenizers::Tokenizer,
}

fn load_npy_vectors(path: &Path) -> Option<Vec<[f32; EMBED_DIM]>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return None;
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let data_start = 10 + header_len;
    if data_start > bytes.len() {
        return None;
    }
    let data = &bytes[data_start..];
    let n = data.len() / (EMBED_DIM * 4);
    let mut vecs = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = [0f32; EMBED_DIM];
        for j in 0..EMBED_DIM {
            let off = (i * EMBED_DIM + j) * 4;
            row[j] = f32::from_le_bytes(data[off..off + 4].try_into().ok()?);
        }
        vecs.push(row);
    }
    Some(vecs)
}

fn embed_query(
    model: &TypedRunnableModel<TypedModel>,
    tokenizer: &tokenizers::Tokenizer,
    query: &str,
) -> Option<[f32; EMBED_DIM]> {
    let encoding = tokenizer.encode(query, true).ok()?;
    let seq_len = encoding.get_ids().len();

    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let attn_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
    let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();

    let ids = tensor1(&input_ids).into_shape(&[1, seq_len]).ok()?;
    let mask = tensor1(&attn_mask).into_shape(&[1, seq_len]).ok()?;
    let types = tensor1(&type_ids).into_shape(&[1, seq_len]).ok()?;

    let result = model.run(tvec![ids.into(), mask.into(), types.into()]).ok()?;
    let hidden_data = result[0].as_slice::<f32>().ok()?;

    // Mean pool weighted by attention mask
    let mask_f: Vec<f32> = encoding
        .get_attention_mask()
        .iter()
        .map(|&x| x as f32)
        .collect();
    let mask_sum: f32 = mask_f.iter().sum();
    if mask_sum == 0.0 {
        return None;
    }

    let mut pooled = [0f32; EMBED_DIM];
    for (tok_idx, &w) in mask_f.iter().enumerate() {
        for dim in 0..EMBED_DIM {
            pooled[dim] += w * hidden_data[tok_idx * EMBED_DIM + dim];
        }
    }
    for v in &mut pooled {
        *v /= mask_sum;
    }

    // L2 normalise
    let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut pooled {
            *v /= norm;
        }
    }

    Some(pooled)
}

fn dot_product_search(
    semantic: &SemanticIndex,
    query_vec: &[f32; EMBED_DIM],
    category: &Option<String>,
    top_k: usize,
) -> Vec<(f32, usize)> {
    let mut scored: Vec<(f32, usize)> = semantic
        .vectors
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            category
                .as_ref()
                .map(|c| semantic.metadata[*i].category == *c)
                .unwrap_or(true)
        })
        .map(|(i, vec)| {
            let dot: f32 = vec.iter().zip(query_vec.iter()).map(|(a, b)| a * b).sum();
            (dot, i)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
}

fn build_semantic_index(index_dir: &Path) -> Option<SemanticIndex> {
    let vectors_path = index_dir.join("vectors.npy");
    let metadata_path = index_dir.join("metadata.json");
    let model_path = index_dir.join("model.onnx");
    let tokenizer_path = index_dir.join("tokenizer.json");

    let vectors = load_npy_vectors(&vectors_path)?;
    let meta_json = std::fs::read_to_string(&metadata_path).ok()?;
    let metadata: Vec<VectorMeta> = serde_json::from_str(&meta_json).ok()?;

    if vectors.len() != metadata.len() {
        eprintln!(
            "knowledge-mcp: vector/metadata count mismatch ({} vs {})",
            vectors.len(),
            metadata.len()
        );
        return None;
    }

    let mut onnx = tract_onnx::onnx()
        .model_for_path(&model_path)
        .map_err(|e| eprintln!("knowledge-mcp: ONNX load failed: {e}"))
        .ok()?;
    let s = onnx.symbols.sym("S");
    let dim1 = 1.to_dim();
    let dim_s = s.to_dim();
    for i in 0..3 {
        onnx.set_input_fact(
            i,
            InferenceFact::dt_shape(i64::datum_type(), &[dim1.clone(), dim_s.clone()]),
        )
        .ok()?;
    }
    let model = onnx
        .into_optimized()
        .and_then(|m| m.into_runnable())
        .map_err(|e| eprintln!("knowledge-mcp: model optimize failed: {e}"))
        .ok()?;

    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).ok()?;

    eprintln!(
        "knowledge-mcp: semantic index loaded ({} vectors, {})",
        vectors.len(),
        model_path.display()
    );

    Some(SemanticIndex {
        vectors,
        metadata,
        model,
        tokenizer,
    })
}

// --- Tools ---

#[macros::mcp_tool(
    name = "search",
    description = "Search the knowledge corpus for Z80, Spectrum, or WASM chunks. Keyword-based — best for exact names like 'LD A' or 'BEEPER'. Use search_semantic for natural language queries.",
    read_only_hint = true
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct SearchTool {
    /// The search query.
    query: String,
    /// Optional category filter (e.g. z80-instruction, wasm-spec).
    category: Option<String>,
}

impl SearchTool {
    fn call(&self, index: &Index) -> Result<CallToolResult, CallToolError> {
        let q = self.query.trim();
        if q.is_empty() {
            return Ok(CallToolResult::text_content(vec!["Query cannot be empty.".into()]));
        }
        let ql = q.to_lowercase();
        let mut scored: Vec<(i32, usize)> = index
            .chunks
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                if let Some(ref cat) = self.category {
                    if c.category != *cat {
                        return None;
                    }
                }
                let mut score = 0i32;
                if c.topic.to_lowercase().contains(&ql) {
                    score += 10;
                }
                if c.stem.to_lowercase().contains(&ql) {
                    score += 5;
                }
                if c.body_lower.contains(&ql) {
                    score += 1;
                }
                if score > 0 { Some((score, i)) } else { None }
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored.truncate(5);

        if scored.is_empty() {
            return Ok(CallToolResult::text_content(vec![
                TextContent::from(format!("No chunks found for query: {q:?}"))
            ]));
        }

        let mut lines = vec![format!("Top {} results for '{q}':\n", scored.len())];
        for (score, i) in &scored {
            let c = &index.chunks[*i];
            let (_, body) = parse_frontmatter(&c.full_text);
            let preview: String = body.chars().take(200).collect();
            let preview = preview.replace('\n', " ");
            let topic = if c.topic.is_empty() { &c.stem } else { &c.topic };
            lines.push(format!(
                "**{}/{}** (score {score})\n  topic: {topic}\n  {preview}\n",
                c.category, c.stem
            ));
        }
        Ok(CallToolResult::text_content(vec![TextContent::from(
            lines.join("\n"),
        )]))
    }
}

#[macros::mcp_tool(
    name = "search_semantic",
    description = "Semantic search the knowledge corpus using sentence embeddings. Better for natural language queries like 'how to play audio on the Spectrum' or 'binary encoding format'. Use keyword search for exact identifiers.",
    read_only_hint = true
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct SearchSemanticTool {
    /// The search query.
    query: String,
    /// Optional category filter (e.g. z80-instruction, wasm-spec).
    category: Option<String>,
}

impl SearchSemanticTool {
    fn call(
        &self,
        semantic: &Option<SemanticIndex>,
        chunks_dir: &Path,
    ) -> Result<CallToolResult, CallToolError> {
        let q = self.query.trim();
        if q.is_empty() {
            return Ok(CallToolResult::text_content(vec!["Query cannot be empty.".into()]));
        }

        let sem = match semantic {
            Some(s) => s,
            None => {
                return Ok(CallToolResult::text_content(vec![TextContent::from(
                    "Semantic search not available. Run `python knowledge/scripts/embed.py` \
                     and `python knowledge/scripts/export_onnx.py` to build the index, \
                     then restart the MCP server."
                        .to_string(),
                )]));
            }
        };

        let query_vec = match embed_query(&sem.model, &sem.tokenizer, q) {
            Some(v) => v,
            None => {
                return Ok(CallToolResult::text_content(vec![TextContent::from(
                    "Failed to embed query.".to_string(),
                )]));
            }
        };

        let results = dot_product_search(sem, &query_vec, &self.category, 5);

        if results.is_empty() {
            return Ok(CallToolResult::text_content(vec![TextContent::from(
                format!("No results for: {q:?}"),
            )]));
        }

        let mut lines = vec![format!("Top {} semantic results for '{q}':\n", results.len())];
        for (score, i) in &results {
            let meta = &sem.metadata[*i];
            let stem = Path::new(&meta.path)
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy();

            // Read chunk body for preview
            let chunk_path = chunks_dir
                .join(&meta.category)
                .join(format!("{stem}.md"));
            let preview = std::fs::read_to_string(&chunk_path)
                .ok()
                .map(|text| {
                    let (_, body) = parse_frontmatter(&text);
                    let p: String = body.chars().take(200).collect();
                    p.replace('\n', " ")
                })
                .unwrap_or_default();

            lines.push(format!(
                "**{}/{}** (similarity {score:.3})\n  topic: {}\n  {preview}\n",
                meta.category, stem, meta.topic
            ));
        }

        Ok(CallToolResult::text_content(vec![TextContent::from(
            lines.join("\n"),
        )]))
    }
}

#[macros::mcp_tool(
    name = "read",
    description = "Read the full content of a specific knowledge chunk by its chunk_id (e.g. z80-instruction/LD).",
    read_only_hint = true
)]
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ReadTool {
    /// Chunk ID in category/filename format (e.g. z80-instruction/LD).
    chunk_id: String,
}

impl ReadTool {
    fn call(&self, index: &Index) -> Result<CallToolResult, CallToolError> {
        let id = self.chunk_id.trim().trim_matches('/');
        let Some((category, stem)) = id.split_once('/') else {
            return Ok(CallToolResult::text_content(vec![
                "chunk_id must be category/filename (e.g. z80-instruction/LD)".into(),
            ]));
        };
        let target = index.chunks_dir.join(category).join(format!("{stem}.md"));
        let resolved = match target.canonicalize() {
            Ok(p) => p,
            Err(_) => {
                return Ok(CallToolResult::text_content(vec![
                    TextContent::from(format!("Chunk not found: {id}"))
                ]));
            }
        };
        let chunks_resolved = match index.chunks_dir.canonicalize() {
            Ok(p) => p,
            Err(_) => {
                return Ok(CallToolResult::text_content(vec![
                    "Chunks directory not found.".into(),
                ]));
            }
        };
        if !resolved.starts_with(&chunks_resolved) {
            return Ok(CallToolResult::text_content(vec![
                TextContent::from(format!("Invalid chunk_id: {id}"))
            ]));
        }
        match std::fs::read_to_string(&resolved) {
            Ok(content) => Ok(CallToolResult::text_content(vec![TextContent::from(content)])),
            Err(_) => Ok(CallToolResult::text_content(vec![
                TextContent::from(format!("Chunk not found: {id}"))
            ])),
        }
    }
}

rust_mcp_sdk::tool_box!(KnowledgeTools, [SearchTool, SearchSemanticTool, ReadTool]);

// --- Handler ---

struct KnowledgeHandler {
    index: Index,
    semantic: Option<SemanticIndex>,
}

#[async_trait]
impl ServerHandler for KnowledgeHandler {
    async fn handle_list_tools_request(
        &self,
        _params: Option<PaginatedRequestParams>,
        _context: &RequestContext,
        _runtime: Arc<dyn McpServer>,
    ) -> Result<ListToolsResult, RpcError> {
        Ok(ListToolsResult {
            tools: KnowledgeTools::tools(),
            cache_scope: ListToolsResultCacheScope::Private,
            result_type: "complete".to_string(),
            ttl_ms: 0,
            meta: None,
            next_cursor: None,
        })
    }

    async fn handle_call_tool_request(
        &self,
        params: CallToolRequestParams,
        _context: &RequestContext,
        _runtime: Arc<dyn McpServer>,
    ) -> Result<ServerResult, CallToolError> {
        let tool = KnowledgeTools::try_from(params).map_err(CallToolError::new)?;
        match tool {
            KnowledgeTools::SearchTool(t) => t.call(&self.index).map(|r| r.into()),
            KnowledgeTools::SearchSemanticTool(t) => t
                .call(&self.semantic, &self.index.chunks_dir)
                .map(|r| r.into()),
            KnowledgeTools::ReadTool(t) => t.call(&self.index).map(|r| r.into()),
        }
    }

    async fn handle_list_resources_request(
        &self,
        _params: Option<PaginatedRequestParams>,
        _context: &RequestContext,
        _runtime: Arc<dyn McpServer>,
    ) -> Result<ListResourcesResult, RpcError> {
        Ok(ListResourcesResult {
            resources: vec![
                Resource {
                    name: "categories".into(),
                    uri: "corpus://categories".into(),
                    description: Some("List of chunk categories with file counts.".into()),
                    mime_type: Some("text/plain".into()),
                    annotations: None,
                    size: None,
                    icons: vec![],
                    meta: None,
                    title: None,
                },
                Resource {
                    name: "stats".into(),
                    uri: "corpus://stats".into(),
                    description: Some("Total chunk count and last rebuild timestamp.".into()),
                    mime_type: Some("text/plain".into()),
                    annotations: None,
                    size: None,
                    icons: vec![],
                    meta: None,
                    title: None,
                },
            ],
            cache_scope: ListResourcesResultCacheScope::Private,
            result_type: "complete".to_string(),
            ttl_ms: 0,
            meta: None,
            next_cursor: None,
        })
    }

    async fn handle_read_resource_request(
        &self,
        params: ReadResourceRequestParams,
        _context: &RequestContext,
        _runtime: Arc<dyn McpServer>,
    ) -> Result<ServerResult, RpcError> {
        let uri = &params.uri;
        if uri == "corpus://categories" {
            let mut lines = vec!["Categories:\n".to_string()];
            if let Ok(entries) = std::fs::read_dir(&self.index.chunks_dir) {
                let mut cats: Vec<_> = entries
                    .flatten()
                    .filter(|e| e.path().is_dir())
                    .collect();
                cats.sort_by_key(|e| e.file_name());
                for cat in cats {
                    let count = std::fs::read_dir(cat.path())
                        .map(|rd| rd.flatten().filter(|f| {
                            f.path().extension().and_then(|e| e.to_str()) == Some("md")
                        }).count())
                        .unwrap_or(0);
                    lines.push(format!(
                        "  {}: {} chunks",
                        cat.file_name().to_string_lossy(),
                        count
                    ));
                }
            }
            return Ok(ReadResourceResult {
                contents: vec![TextResourceContents {
                    text: lines.join("\n"),
                    uri: uri.clone(),
                    mime_type: Some("text/plain".into()),
                    meta: None,
                }
                .into()],
                cache_scope: ReadResourceResultCacheScope::Private,
                result_type: "complete".to_string(),
                ttl_ms: 0,
                meta: None,
            }
            .into());
        }
        if uri == "corpus://stats" {
            let total = self.index.chunks.len();
            let semantic_str = if self.semantic.is_some() {
                "available"
            } else {
                "not loaded (run embed.py + export_onnx.py)"
            };
            let mut latest: Option<SystemTime> = None;
            for cat in std::fs::read_dir(&self.index.chunks_dir).into_iter().flatten().flatten() {
                if !cat.path().is_dir() {
                    continue;
                }
                for f in std::fs::read_dir(cat.path()).into_iter().flatten().flatten() {
                    if let Ok(meta) = f.metadata() {
                        if let Ok(mt) = meta.modified() {
                            latest = Some(latest.map_or(mt, |l: SystemTime| l.max(mt)));
                        }
                    }
                }
            }
            let last_str = latest
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|d| {
                    let secs = d.as_secs();
                    format!("{secs} (unix epoch)")
                })
                .unwrap_or_else(|| "unknown".into());
            let text = format!(
                "Total chunks: {total}\nSemantic search: {semantic_str}\nLast rebuild: {last_str}"
            );
            return Ok(ReadResourceResult {
                contents: vec![TextResourceContents {
                    text,
                    uri: uri.clone(),
                    mime_type: Some("text/plain".into()),
                    meta: None,
                }
                .into()],
                cache_scope: ReadResourceResultCacheScope::Private,
                result_type: "complete".to_string(),
                ttl_ms: 0,
                meta: None,
            }
            .into());
        }
        Err(RpcError::invalid_request()
            .with_message(format!("No resource found for '{uri}'.")))
    }
}

// --- Main ---

#[tokio::main]
async fn main() -> SdkResult<()> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let chunks_dir = cwd.join("knowledge").join("chunks");
    let index_dir = cwd.join("knowledge").join("index");

    let chunks = build_index(&chunks_dir);
    eprintln!("knowledge-mcp: loaded {} keyword chunks", chunks.len());

    let index = Index {
        chunks,
        chunks_dir,
    };

    let semantic = build_semantic_index(&index_dir);

    let server_details = ServerDetails {
        server_info: Implementation {
            name: "knowledge-corpus".into(),
            version: "0.2.0".into(),
            title: Some("Knowledge Corpus".into()),
            description: Some("Search and read Z80/Spectrum/WASM knowledge chunks.".into()),
            icons: vec![],
            website_url: None,
        },
        capabilities: ServerCapabilities {
            tools: Some(ServerCapabilitiesTools {
                list_changed: Some(false),
            }),
            resources: Some(ServerCapabilitiesResources {
                list_changed: Some(false),
                subscribe: Some(false),
            }),
            ..Default::default()
        },
        instructions: None,
        meta: None,
    };

    let transport = StdioTransport::new(TransportOptions::default())?;
    let handler = KnowledgeHandler { index, semantic }.to_mcp_server_handler();
    let server = server_runtime::create_server(McpServerOptions {
        transport,
        handler,
        server_details,
        message_observer: None,
    });
    server.start().await
}
