//! PRIVATE actual resident owner-thread adapter. No loading, listener, automatic
//! launch or qualification. Caller supplies the still-live model/comm and an
//! actual checked bootstrap completion. Every error requires all8 teardown.
use super::{
    gpu_collectives::GpuCollectives,
    gpu_model_chunk::{CheckedModelChunk, GpuModelChunk},
    gpu_request_session::CheckedRequestCompletion,
    native_api_contract::CpuBridge,
    native_http_actor::{NativeFailure, ResidentControl, ResidentInbox},
    native_request_dispatch::{HeadDispatch, PeerDispatch},
    native_request_executor,
    native_response_bridge::ResponseAuthorityBinding,
    request_session_release::RequestReleaseConfig,
};
use anyhow::{ensure, Context, Result};
use std::io::Write;
fn completion_evidence(request: &str, sequence: usize, c: &CheckedRequestCompletion) -> Result<()> {
    let value = serde_json::json!({"event":"RESIDENT_CHECKED_REQUEST_COMPLETE","rank":c.rank(),"request_id":request,"sequence":sequence,"checked_request_completion":c,"benchmark":false,"coding_or_tool_quality_scored":false});
    let mut output = std::io::stdout().lock();
    serde_json::to_writer(&mut output, &value)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}
use std::{
    sync::atomic::{AtomicBool, Ordering},
    thread::{self, ThreadId},
};

/// Fixed independently supervised expected deployment metadata. It is compared
/// with actual opaque bootstrap/model/session metadata before accepting HTTP.
#[derive(Clone, Debug)]
pub struct ResidentDeployment {
    pub global_checkpoint_geometry_checksum_sha256: String,
    pub execution_snapshot_sha256: String,
    pub engine_binary_sha256: String,
    pub request_release_template: RequestReleaseConfig,
}
impl ResidentDeployment {
    fn check(
        &self,
        previous: &CheckedRequestCompletion,
        model: &GpuModelChunk,
        comm: &GpuCollectives,
    ) -> Result<()> {
        self.request_release_template.validate()?;
        let b = previous.binding();
        let m = model.identity();
        ensure!(
            previous.rank() == comm.rank()
                && previous.all_rank_prefix_agreement()
                && previous.transport_lifetime_release_confirmed()
                && previous.completion_owner() == comm.completion_identity()
                && previous.actual_epoch() == comm.expected_epoch()
                && previous.program_sha256() == comm.program_digest_hex()
                && Some(previous.plan_sha256()) == comm.request_plan_digest(),
            "resident adapter requires current actual checked bootstrap/request boundary"
        );
        ensure!(
            b.global_checkpoint_geometry_checksum_sha256
                == self.global_checkpoint_geometry_checksum_sha256
                && b.execution_snapshot_sha256 == self.execution_snapshot_sha256
                && b.engine_binary_sha256 == self.engine_binary_sha256
                && m.global_checkpoint_geometry_checksum_sha256
                    == self.global_checkpoint_geometry_checksum_sha256
                && m.runtime.execution_snapshot_sha256 == self.execution_snapshot_sha256
                && m.runtime.engine_binary_sha256 == self.engine_binary_sha256
                && m.rank == comm.rank(),
            "resident actual deployment differs from supervised model/ELF/snapshot identity"
        );
        Ok(())
    }
    fn release(&self, port: u16) -> Result<RequestReleaseConfig> {
        let mut c = self.request_release_template.clone();
        c.port = port;
        c.validate()?;
        Ok(c)
    }
}
fn owner(expected: ThreadId) -> Result<()> {
    ensure!(
        thread::current().id() == expected,
        "native model/comm moved off initializing owner thread"
    );
    Ok(())
}

/// Blocking receive + synchronous GPU work stay on the caller's CUDA thread.
/// `parser_runtime` is used only to drive the official CPU buffered parser on
/// this same thread; the HTTP runtime/listener is owned separately. Do not call
/// this blocking loop as a spawned multithread async task.
#[allow(clippy::too_many_arguments)]
pub fn run_head_worker(
    model: &mut GpuModelChunk,
    comm: &mut GpuCollectives,
    mut previous: CheckedRequestCompletion,
    deployment: ResidentDeployment,
    mut dispatch: HeadDispatch,
    bridge: &CpuBridge,
    mut inbox: ResidentInbox,
    control: ResidentControl,
    parser_runtime: &tokio::runtime::Runtime,
    mut observe: impl FnMut(&str, usize, &CheckedModelChunk) -> Result<()>,
) -> Result<CheckedRequestCompletion> {
    let thread = thread::current().id();
    ensure!(comm.rank() == 0, "head worker rank0 only");
    deployment.check(&previous, model, comm)?;
    control.mark_ready_after_coordinated_native_reset();
    while let Some(work) = inbox.blocking_recv() {
        owner(thread)?;
        let (chat, response_id, admitted, deadline, _raw_request, cancelled, reply, mut streaming) =
            work.into_parts_with_stream();
        if !control.ready() {
            let _ = reply.failed(NativeFailure {
                class: "resident_not_ready".into(),
                detail: "native owner requires coordinated all8 recovery".into(),
                all_eight_cleanup_confirmed: false,
                cleanup_elapsed_seconds: None,
            });
            dispatch.poison();
            comm.abort_session();
            anyhow::bail!("request reached blocked resident owner");
        }
        let outcome = (|| {
            ensure!(
                !cancelled.load(Ordering::Acquire) && std::time::Instant::now() < deadline,
                "HTTP request cancelled/expired before native dispatch"
            );
            deployment.check(&previous, model, comm)?;
            ensure!(
                chat.semantics().stream == streaming.is_some(),
                "native delivery/request stream modes differ"
            );
            let plan = if streaming.is_some() {
                native_request_executor::next_stream_plan(&chat, &previous)?
            } else {
                native_request_executor::next_chat_plan(&chat, &previous)?
            };
            let sequence = dispatch.begin(&chat, &plan, &previous, &response_id, admitted)?;
            let release = deployment.release(dispatch.release_port(sequence)?)?;
            let forced = chat.forced_named_cursor()?;
            let completed = if let Some(publisher) = streaming.as_mut() {
                native_request_executor::execute_next_request_stream(
                    model,
                    comm,
                    &previous,
                    plan.clone(),
                    release,
                    admitted,
                    &cancelled,
                    |proof| {
                        owner(thread)?;
                        observe(&response_id, sequence, proof)
                    },
                    forced,
                    |token| {
                        owner(thread)?;
                        publisher.accepted(token)
                    },
                )?
            } else {
                native_request_executor::execute_next_request_controlled(
                    model,
                    comm,
                    &previous,
                    plan.clone(),
                    release,
                    admitted,
                    &cancelled,
                    |proof| {
                        owner(thread)?;
                        observe(&response_id, sequence, proof)
                    },
                    forced,
                    |_| Ok(()),
                )?
            };
            owner(thread)?;
            completion_evidence(&response_id, sequence, &completed)?;
            dispatch.finish(&completed, admitted)?;
            ensure!(
                !cancelled.load(Ordering::Acquire) && std::time::Instant::now() < deadline,
                "request cancelled/expired after native completion control"
            );
            let actual = model.identity();
            let expected = ResponseAuthorityBinding {
                global_checkpoint_geometry_checksum_sha256: deployment
                    .global_checkpoint_geometry_checksum_sha256
                    .clone(),
                execution_snapshot_sha256: deployment.execution_snapshot_sha256.clone(),
                engine_binary_sha256: deployment.engine_binary_sha256.clone(),
                head_manifest_sha256: actual.manifest_sha256,
                frequency: plan.binding().frequency.clone(),
                lane: plan.binding().lane,
                completion_owner: comm.completion_identity(),
                program_sha256: comm.program_digest_hex(),
            };
            let parsed = if let Some(publisher) = streaming.as_mut() {
                let finish = super::native_response_bridge::checked_stream_finish(
                    &chat,
                    &plan,
                    &expected,
                    &completed,
                    response_id.clone(),
                )?;
                publisher.finish(finish)?;
                None
            } else {
                Some(
                    parser_runtime.block_on(native_request_executor::parse_actual_response(
                        bridge,
                        chat,
                        &plan,
                        &expected,
                        &completed,
                        response_id.clone(),
                    )),
                )
            };
            owner(thread)?;
            Ok((completed, parsed))
        })();
        match outcome {
            Ok((completed, parsed)) => {
                let passed = parsed.as_ref().is_none_or(|p| p.passed());
                let error = parsed.as_ref().and_then(|p| p.error()).map(str::to_owned);
                if let Some(parsed) = parsed {
                    if let Err(e) = reply.checked(parsed) {
                        control.block_until_coordinated_native_recovery();
                        dispatch.poison();
                        comm.abort_session();
                        return Err(e);
                    }
                }
                previous = completed;
                if !passed
                    || cancelled.load(Ordering::Acquire)
                    || (streaming.as_ref().is_none_or(|s| !s.finished_on_time())
                        && std::time::Instant::now() >= deadline)
                {
                    control.block_until_coordinated_native_recovery();
                    dispatch.poison();
                    comm.abort_session();
                    anyhow::bail!("buffered native request failed/detached; coordinated teardown required: {}",error.unwrap_or_else(||"cancelled/deadline".into()));
                }
            }
            Err(e) => {
                control.block_until_coordinated_native_recovery();
                dispatch.poison();
                comm.abort_session();
                let detail = format!("{e:#}");
                if let Some(publisher) = streaming.as_mut() {
                    let _ = publisher.failed(detail.clone());
                }
                let retained = reply.failed(NativeFailure {
                    class: "native_resident_request_failed".into(),
                    detail: detail.clone(),
                    all_eight_cleanup_confirmed: false,
                    cleanup_elapsed_seconds: None,
                });
                if let Err(log) = retained {
                    return Err(e.context(format!("worker failure evidence error: {log:#}")));
                }
                return Err(e);
            }
        }
    }
    control.block_until_coordinated_native_recovery();
    dispatch.stop("HTTP actor closed after checked idle boundary")?;
    Ok(previous)
}

/// Rank1..7 independently prepare the exact admitted body on their own native
/// owner threads. They never accept terminal JSON or execute fixture tools.
/// Head disconnect/cancel does not promise instant GPU preemption: any error
/// poisons transport and parent independently tears down the full cluster.
#[allow(clippy::too_many_arguments)]
pub fn run_peer_worker(
    model: &mut GpuModelChunk,
    comm: &mut GpuCollectives,
    mut previous: CheckedRequestCompletion,
    deployment: ResidentDeployment,
    mut dispatch: PeerDispatch,
    bridge: &CpuBridge,
    mut observe: impl FnMut(&str, usize, &CheckedModelChunk) -> Result<()>,
) -> Result<CheckedRequestCompletion> {
    let thread = thread::current().id();
    ensure!((1..8).contains(&comm.rank()), "worker rank1..7 only");
    deployment.check(&previous, model, comm)?;
    let outcome = (|| {
        while let Some(request) = dispatch.receive(bridge, &previous)? {
            owner(thread)?;
            deployment.check(&previous, model, comm)?;
            let (chat, plan, response_id, sequence, admitted) = request.into_parts();
            let release = deployment.release(dispatch.release_port(sequence)?)?;
            // HTTP cancellation lives on rank0. Its native abort and the
            // independent all8 supervisor provide teardown; this local false
            // flag never purports to receive an instantaneous peer broadcast.
            let local_cancel = AtomicBool::new(false);
            let forced = chat.forced_named_cursor()?;
            let completed = native_request_executor::execute_next_request_controlled(
                model,
                comm,
                &previous,
                plan,
                release,
                admitted,
                &local_cancel,
                |proof| {
                    owner(thread)?;
                    observe(&response_id, sequence, proof)
                },
                forced,
                |_| Ok(()),
            )?;
            completion_evidence(&response_id, sequence, &completed)?;
            dispatch.finish(&completed, admitted)?;
            previous = completed;
        }
        Ok(())
    })();
    if let Err(e) = outcome {
        dispatch.poison();
        comm.abort_session();
        return Err(e);
    }
    Ok(previous)
}
