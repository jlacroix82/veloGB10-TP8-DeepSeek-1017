//! DeepSeek-V4.1-Flash-1017 metadata and CPU planning contracts.
//! No inference implementation is registered by these modules.

pub mod attention_math;
pub mod attention_state;
pub mod bf16_collectives;
pub mod cache_quant;
pub mod chunk_plan;
pub mod collectives;
pub mod compression;
pub mod config;
pub mod engram;
pub mod engram_math;
pub mod gpu_attention;
pub mod gpu_attention_block;
pub mod gpu_attention_chunk;
pub mod gpu_batch_engram;
pub mod gpu_bf16_collectives;
pub mod gpu_block;
pub mod gpu_block_chunk;
pub mod gpu_cache_quant;
pub mod gpu_chunk_compressor;
pub mod gpu_chunk_store;
pub mod gpu_collectives;
pub mod gpu_compression;
pub mod gpu_dense;
pub mod gpu_engram;
pub mod gpu_engram_block;
pub mod gpu_engram_chunk;
pub mod gpu_experts;
pub mod gpu_gemm;
pub mod gpu_hc;
mod gpu_hc_graph;
#[doc(hidden)]
pub use gpu_hc_graph::check_device_replay as check_hc_graph_device_replay;
pub mod gpu_index_block;
pub mod gpu_index_select;
pub mod gpu_kv_store;
pub mod gpu_model;
pub mod gpu_model_weights;
pub mod gpu_module;
pub mod gpu_moe;
pub mod gpu_moe_block;
pub mod gpu_numerics;
pub mod gpu_position;
pub mod gpu_reader;
pub mod gpu_status_cut;
pub mod gpu_text;
pub mod gpu_text_batch;
pub mod gpu_text_head;
pub mod gpu_text_head_batch;
pub mod gpu_weights;
pub mod hc;
pub mod inventory;
pub mod kv_store;
pub mod model_session_plan;
pub mod moe;
pub mod native_api_contract;
pub mod lab_network;
pub mod numerics;
pub mod plan;
pub mod position;
pub mod protocol;
pub mod reader;
pub mod scheduler;
pub mod weights;

pub use config::{CapabilitySummary, Dsv41Config};
pub mod gpu_model_chunk;
pub mod gpu_request_session;
pub mod native_http_actor;
pub mod native_request_executor;
pub mod native_response_bridge;
pub mod request_session_release;
pub mod request_session_state;

pub mod gpu_chunk_source_indices;
pub mod native_request_dispatch;
pub mod native_resident_worker;

pub mod native_streaming;
pub mod tool_constraint;

pub mod gpu_tool_constraint;
pub mod batch_attention_layout;
pub mod batch_plan;
pub mod gpu_batch_attention;
pub mod gpu_batch_store;

pub mod batch_compression_plan;
pub mod gpu_batch_attention_block;
pub mod gpu_batch_compressor;

pub mod gpu_batch_block;

pub mod gpu_batch_model;

pub mod gpu_batch_text_head;

pub mod atomic_request_admission;
pub mod batch_request_plan;
pub mod batch_request_release;
pub mod gpu_batch_request_session;
pub mod native_batch_response;
pub mod native_batch_dispatch;
pub mod native_batch_http;
