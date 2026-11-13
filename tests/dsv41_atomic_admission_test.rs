//! CPU-only integration harness for an intentionally unregistered private module.
pub use gb10_inference::dsv41::{
    config, model_session_plan, native_api_contract, position, scheduler,
};
#[path = "../src/dsv41/atomic_request_admission.rs"]
mod atomic_request_admission;
