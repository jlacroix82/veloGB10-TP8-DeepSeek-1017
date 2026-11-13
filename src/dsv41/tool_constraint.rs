//! Request-bound, fail-closed native-token constraint for a named function.
//!
//! The pinned toolkit filters named choices to one Required tool and prefills
//! the calls block. This module enforces that continuation before greedy
//! selection. Ordinary bytes cannot impersonate the native DSML token. It
//! permits only complete JSON-valued/schema-valid parameters and at least one
//! invocation of the selected function before calls-close/EOS. This is a new
//! explicit constrained-decoding policy, not unconstrained-prompt equivalence.
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};

pub const VOCAB: usize = 129280;
pub const EOS: u32 = 1;
pub const DSML: u32 = 128825;
pub const POLICY: &str = "dsv41-forced-named-native-token-json-v1";
// Actual native output budget4096 × verified maximum token byte length128.
// This bound cannot shorten a legal runtime output allowance.
const MAX_VALUE_BYTES: usize = 4096 * 128;
const MAX_PROPERTIES: usize = 32;
const MAX_CALLS: usize = 4096;
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn ids_hash(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Atom {
    Byte(u8),
    Dsml,
    Eos,
}
#[derive(Default)]
struct Node {
    children: BTreeMap<Atom, usize>,
    ids: Vec<u32>,
}
/// Verified ByteLevel bytes, not lossy per-token UTF8 decoding. Every added
/// control token is excluded except the precise DSML/EOS IDs used by the AST.
#[derive(Default)]
pub struct ConstraintVocabulary {
    nodes: Vec<Node>,
    atoms: Vec<Option<Vec<Atom>>>,
    sha256: String,
}
impl std::fmt::Debug for ConstraintVocabulary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstraintVocabulary")
            .field("sha256", &self.sha256)
            .field("trie_nodes", &self.nodes.len())
            .finish()
    }
}
impl ConstraintVocabulary {
    pub fn from_verified_json(bytes: &[u8], expected: &str) -> Result<Arc<Self>> {
        ensure!(
            hash(bytes) == expected,
            "constraint tokenizer checksum differs"
        );
        let j: Value = serde_json::from_slice(bytes)?;
        ensure!(
            j["decoder"]["type"] == "ByteLevel" && j["model"]["type"] == "BPE",
            "only the verified native ByteLevel tokenizer is supported"
        );
        let mut reverse = BTreeMap::new();
        let visible: Vec<u16> = (33..=126).chain(161..=172).chain(174..=255).collect();
        for &b in &visible {
            reverse.insert(char::from_u32(b as u32).unwrap(), b as u8);
        }
        let mut n = 0;
        for b in 0..=255u16 {
            if !visible.contains(&b) {
                reverse.insert(char::from_u32(256 + n).unwrap(), b as u8);
                n += 1;
            }
        }
        let added = j["added_tokens"]
            .as_array()
            .context("native added token list missing")?;
        let excluded: BTreeSet<u32> = added
            .iter()
            .map(|v| {
                v["id"]
                    .as_u64()
                    .and_then(|v| u32::try_from(v).ok())
                    .context("native added token ID")
            })
            .collect::<Result<_>>()?;
        ensure!(
            added
                .iter()
                .any(|v| v["id"] == DSML && v["content"] == "｜DSML｜")
                && added
                    .iter()
                    .any(|v| v["id"] == EOS && v["content"] == "<｜end▁of▁sentence｜>"),
            "native DSML/EOS identity differs"
        );
        let mut atoms = vec![None; VOCAB];
        for (spelling, id) in j["model"]["vocab"]
            .as_object()
            .context("native BPE vocabulary missing")?
        {
            let id = usize::try_from(id.as_u64().context("BPE ID not integer")?)?;
            ensure!(id < VOCAB, "native vocabulary ID exceeds fixed head");
            if excluded.contains(&(id as u32)) {
                continue;
            }
            let a = spelling
                .chars()
                .map(|c| {
                    reverse
                        .get(&c)
                        .copied()
                        .map(Atom::Byte)
                        .context("non-ByteLevel base vocabulary spelling")
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                !a.is_empty() && a.len() <= 128,
                "bounded native token bytes"
            );
            atoms[id] = Some(a);
        }
        atoms[DSML as usize] = Some(vec![Atom::Dsml]);
        atoms[EOS as usize] = Some(vec![Atom::Eos]);
        let mut nodes = vec![Node::default()];
        for (id, a) in atoms.iter().enumerate() {
            if let Some(a) = a {
                let mut at = 0;
                for symbol in a {
                    let child = if let Some(&c) = nodes[at].children.get(symbol) {
                        c
                    } else {
                        let c = nodes.len();
                        nodes.push(Node::default());
                        nodes[at].children.insert(*symbol, c);
                        c
                    };
                    at = child;
                }
                nodes[at].ids.push(id as u32);
            }
        }
        Ok(Arc::new(Self {
            nodes,
            atoms,
            sha256: expected.into(),
        }))
    }
    pub fn tokenizer_sha256(&self) -> &str {
        &self.sha256
    }
}
fn safe_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.'))
}
fn permits_kind(schema: &Value, kind: &str) -> bool {
    let Some(t) = schema.get("type") else {
        return true;
    };
    let permits = |t: &str| t == kind || (kind == "number" && t == "integer");
    t.as_str().is_some_and(permits)
        || t.as_array()
            .is_some_and(|a| a.iter().filter_map(Value::as_str).any(permits))
}
fn json_kind(bytes: &[u8]) -> Option<&'static str> {
    bytes.iter().copied().find(|b| !ws(*b)).map(|b| match b {
        b'"' => "string",
        b'{' => "object",
        b'[' => "array",
        b't' | b'f' => "boolean",
        b'n' => "null",
        _ => "number",
    })
}
fn no_references(v: &Value) -> bool {
    match v {
        Value::Object(m) => {
            !m.contains_key("$ref")
                && !m.contains_key("$dynamicRef")
                && m.values().all(no_references)
        }
        Value::Array(a) => a.iter().all(no_references),
        _ => true,
    }
}
pub struct NamedToolSpec {
    name: String,
    schema: Value,
    validator: jsonschema::Validator,
    properties: BTreeMap<String, jsonschema::Validator>,
    required: BTreeSet<String>,
    sha256: String,
}
impl std::fmt::Debug for NamedToolSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedToolSpec")
            .field("name", &self.name)
            .field("schema_sha256", &self.sha256)
            .finish()
    }
}
impl NamedToolSpec {
    pub fn new(name: String, schema: Value) -> Result<Arc<Self>> {
        ensure!(
            safe_name(&name),
            "forced tool name has unsupported DSML attribute syntax"
        );
        let m = schema
            .as_object()
            .context("forced tool parameters must be object schema")?;
        ensure!(
            m.get("type") == Some(&Value::String("object".into()))
                && m.get("additionalProperties") == Some(&Value::Bool(false)),
            "forced constraint currently requires a closed object schema"
        );
        ensure!(
            no_references(&schema),
            "external/recursive schema references are unsupported"
        );
        // Top-level combinators/dependencies could require optional parameters
        // in ways this bounded attribute-name grammar cannot prove prefix-safe.
        for k in m.keys() {
            ensure!(
                [
                    "type",
                    "properties",
                    "required",
                    "additionalProperties",
                    "description",
                    "title",
                    "$schema"
                ]
                .contains(&k.as_str()),
                "unsupported forced top-level schema keyword:{k}"
            );
        }
        let p = m
            .get("properties")
            .and_then(Value::as_object)
            .context("forced schema properties missing")?;
        ensure!(
            p.len() <= MAX_PROPERTIES,
            "forced schema exceeds32 properties"
        );
        let mut properties = BTreeMap::new();
        for (name, s) in p {
            ensure!(safe_name(name), "unsupported DSML argument name");
            let property = s
                .as_object()
                .context("forced property schema must be object")?;
            ensure!(property.keys().all(|k|["type","description","title"].contains(&k.as_str())),
                "forced property constraint currently supports arbitrary JSON (empty schema) or primitive type-only schemas; nested/combinator/enum/range schemas require a qualified schema-prefix solver");
            properties.insert(
                name.clone(),
                jsonschema::validator_for(s)
                    .map_err(|e| anyhow::anyhow!("invalid property schema:{e}"))?,
            );
        }
        let mut required = BTreeSet::new();
        if let Some(r) = m.get("required") {
            for r in r.as_array().context("required must be array")? {
                let name = r.as_str().context("required name must be string")?;
                ensure!(
                    p.contains_key(name) && required.insert(name.into()),
                    "unknown/duplicate required argument"
                );
            }
        }
        let validator = jsonschema::validator_for(&schema)
            .map_err(|e| anyhow::anyhow!("invalid named schema:{e}"))?;
        let sha256 = hash(&serde_json::to_vec(&(POLICY, &name, &schema))?);
        Ok(Arc::new(Self {
            name,
            schema,
            validator,
            properties,
            required,
            sha256,
        }))
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn schema(&self) -> &Value {
        &self.schema
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn validate_arguments(&self, args: &Value) -> Result<()> {
        ensure!(
            self.validator.is_valid(args),
            "forced named arguments violate the source tool schema"
        );
        Ok(())
    }
}
#[derive(Clone, Debug)]
enum After {
    Invoke,
    Parameter(String, bool),
    ParameterEnd,
    InvokeEnd,
    CallsEnd,
}
#[derive(Clone, Debug)]
enum Stage {
    BetweenCalls,
    BetweenParameters,
    Literal {
        symbols: Arc<Vec<Atom>>,
        at: usize,
        after: After,
    },
    Value {
        key: String,
        string: bool,
        bytes: Vec<u8>,
    },
    AfterCalls,
    End,
}
#[derive(Clone, Debug)]
struct Machine {
    stage: Stage,
    arguments: Map<String, Value>,
    calls: usize,
    completed_values: Option<(String, Value)>,
}
fn ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\r' | b'\t')
}
fn utf8_prefix(bytes: &[u8]) -> bool {
    match std::str::from_utf8(bytes) {
        Ok(_) => true,
        Err(e) => e.error_len().is_none(),
    }
}
fn json_prefix(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return true;
    }
    match serde_json::from_slice::<Value>(bytes) {
        Ok(_) => true,
        Err(e) => e.is_eof(),
    }
}
fn sequence(prefix: &str, dsml: bool, suffix: &str) -> Arc<Vec<Atom>> {
    let mut a = prefix.bytes().map(Atom::Byte).collect::<Vec<_>>();
    if dsml {
        a.push(Atom::Dsml);
    }
    a.extend(suffix.bytes().map(Atom::Byte));
    Arc::new(a)
}
impl Machine {
    fn initial() -> Self {
        Self {
            stage: Stage::BetweenCalls,
            arguments: Map::new(),
            calls: 0,
            completed_values: None,
        }
    }
    fn literal(&self, symbols: Arc<Vec<Atom>>, after: After, first: Atom) -> Option<Self> {
        if symbols.first() != Some(&first) {
            return None;
        }
        let mut m = self.clone();
        m.stage = Stage::Literal {
            symbols,
            at: 1,
            after,
        };
        Some(m)
    }
    fn apply(&mut self, after: After) -> Option<()> {
        self.stage = match after {
            After::Invoke => {
                self.arguments.clear();
                Stage::BetweenParameters
            }
            After::Parameter(key, string) => Stage::Value {
                key,
                string,
                bytes: Vec::new(),
            },
            After::ParameterEnd => {
                let (k, v) = self.completed_values.take()?;
                if self.arguments.insert(k, v).is_some() {
                    return None;
                }
                Stage::BetweenParameters
            }
            After::InvokeEnd => {
                self.calls += 1;
                Stage::BetweenCalls
            }
            After::CallsEnd => Stage::AfterCalls,
        };
        Some(())
    }
    fn feed(&self, a: Atom, spec: &NamedToolSpec) -> Vec<Self> {
        let mut out = Vec::new();
        match &self.stage {
            Stage::BetweenCalls => {
                if let Atom::Byte(b) = a {
                    if ws(b) {
                        out.push(self.clone());
                    }
                }
                if self.calls < MAX_CALLS {
                    if let Some(m) = self.literal(
                        sequence("<", true, &format!(" invoke name=\"{}\">", spec.name)),
                        After::Invoke,
                        a,
                    ) {
                        out.push(m);
                    }
                }
                if self.calls > 0 {
                    if let Some(m) =
                        self.literal(sequence("</", true, " calls>"), After::CallsEnd, a)
                    {
                        out.push(m);
                    }
                }
            }
            Stage::BetweenParameters => {
                if let Atom::Byte(b) = a {
                    if ws(b) {
                        out.push(self.clone());
                    }
                }
                for name in spec
                    .properties
                    .keys()
                    .filter(|n| !self.arguments.contains_key(*n))
                {
                    for string in [false, true] {
                        let property = &spec.schema["properties"][name];
                        if string && !permits_kind(property, "string") {
                            continue;
                        }
                        if !string
                            && !["null", "boolean", "number", "array", "object"]
                                .iter()
                                .any(|k| permits_kind(property, k))
                        {
                            continue;
                        }
                        if let Some(m) = self.literal(
                            sequence(
                                "<",
                                true,
                                &format!(
                                    " parameter name=\"{name}\" string=\"{}\">",
                                    if string { "true" } else { "false" }
                                ),
                            ),
                            After::Parameter(name.clone(), string),
                            a,
                        ) {
                            out.push(m);
                        }
                    }
                }
                if spec.required.iter().all(|k| self.arguments.contains_key(k))
                    && spec
                        .validator
                        .is_valid(&Value::Object(self.arguments.clone()))
                {
                    if let Some(m) =
                        self.literal(sequence("</", true, " invoke>"), After::InvokeEnd, a)
                    {
                        out.push(m);
                    }
                }
            }
            Stage::Literal { symbols, at, after } => {
                if symbols.get(*at) == Some(&a) {
                    let mut m = self.clone();
                    if *at + 1 == symbols.len() {
                        if m.apply(after.clone()).is_some() {
                            out.push(m);
                        }
                    } else {
                        m.stage = Stage::Literal {
                            symbols: symbols.clone(),
                            at: at + 1,
                            after: after.clone(),
                        };
                        out.push(m);
                    }
                }
            }
            Stage::Value { key, string, bytes } => {
                if let Atom::Byte(b) = a {
                    if bytes.len() < MAX_VALUE_BYTES {
                        let mut next = bytes.clone();
                        next.push(b);
                        let valid = if *string {
                            b >= 32 || matches!(b, b'\n' | b'\r' | b'\t')
                        } else {
                            json_kind(&next).is_none_or(|k| {
                                k != "string" && permits_kind(&spec.schema["properties"][key], k)
                            }) && json_prefix(&next)
                                // Whitespace ends a JSON number. A fractional
                                // prefix may still gain an exponent, but a
                                // whitespace-committed wrong primitive cannot.
                                && !(ws(b) && serde_json::from_slice::<Value>(&next).ok()
                                    .is_some_and(|v|!spec.properties[key].is_valid(&v)))
                        };
                        if valid && (!*string || utf8_prefix(&next)) {
                            let mut m = self.clone();
                            m.stage = Stage::Value {
                                key: key.clone(),
                                string: *string,
                                bytes: next,
                            };
                            out.push(m);
                        }
                    }
                }
                let parsed = if *string {
                    std::str::from_utf8(bytes)
                        .ok()
                        .map(|s| Value::String(s.into()))
                } else {
                    serde_json::from_slice::<Value>(bytes).ok()
                };
                if let Some(value) = parsed {
                    if spec.properties[key].is_valid(&value) {
                        if let Some(mut m) = self.literal(
                            sequence("</", true, " parameter>"),
                            After::ParameterEnd,
                            a,
                        ) {
                            m.completed_values = Some((key.clone(), value));
                            out.push(m);
                        }
                    }
                }
            }
            Stage::AfterCalls => match a {
                Atom::Byte(b) if ws(b) => out.push(self.clone()),
                Atom::Eos => {
                    let mut m = self.clone();
                    m.stage = Stage::End;
                    out.push(m);
                }
                _ => {}
            },
            Stage::End => {}
        }
        out
    }
}
fn advance(states: &[Machine], symbols: &[Atom], spec: &NamedToolSpec) -> Vec<Machine> {
    let mut s = states.to_vec();
    for &a in symbols {
        s = s.iter().flat_map(|m| m.feed(a, spec)).collect();
        if s.is_empty() {
            break;
        }
    }
    s
}
#[derive(Clone, Debug, Serialize)]
pub struct ConstraintIdentity {
    pub policy: &'static str,
    pub named_tool: String,
    pub schema_sha256: String,
    pub request_sha256: String,
    pub tokenizer_sha256: String,
    pub prompt_ids_sha256: String,
    pub generated_prefix_sha256: String,
    pub generated_prefix_tokens: usize,
    pub allowed_native_tokens: usize,
    pub mask_sha256: String,
    pub build_cpu_millis: f64,
    pub constrained_greedy_before_local_maxloc: bool,
}
/// Cannot be constructed from JSON or arbitrary bits by a caller. Its binding
/// is checked against the actual model/request prefix before any use.
pub struct AllowedNativeTokens {
    bits: Vec<u32>,
    identity: ConstraintIdentity,
}
impl AllowedNativeTokens {
    pub fn words(&self) -> &[u32] {
        &self.bits
    }
    pub fn identity(&self) -> &ConstraintIdentity {
        &self.identity
    }
    pub fn allows(&self, id: u32) -> bool {
        (id as usize) < VOCAB && self.bits[id as usize / 32] & (1 << (id % 32)) != 0
    }
    pub fn check_binding(&self, request: &str, prompt: &[u32], generated: &[u32]) -> Result<()> {
        ensure!(
            self.identity.request_sha256 == request
                && self.identity.prompt_ids_sha256 == ids_hash(prompt)
                && self.identity.generated_prefix_sha256 == ids_hash(generated)
                && self.identity.generated_prefix_tokens == generated.len(),
            "constraint mask belongs to a different request/native prefix"
        );
        Ok(())
    }
}
pub struct ForcedNamedCursor {
    vocab: Arc<ConstraintVocabulary>,
    spec: Arc<NamedToolSpec>,
    request: String,
    prompt_sha: String,
    generated: Vec<u32>,
    states: Vec<Machine>,
}
impl ForcedNamedCursor {
    pub fn new(
        vocab: Arc<ConstraintVocabulary>,
        spec: Arc<NamedToolSpec>,
        request: String,
        prompt: &[u32],
    ) -> Result<Self> {
        ensure!(
            request.len() == 64
                && request.bytes().all(|b| b.is_ascii_hexdigit())
                && !prompt.is_empty(),
            "constraint exact request/prompt identity"
        );
        Ok(Self {
            vocab,
            spec,
            request,
            prompt_sha: ids_hash(prompt),
            generated: Vec::new(),
            states: vec![Machine::initial()],
        })
    }
    pub fn generated_ids(&self) -> &[u32] {
        &self.generated
    }
    pub fn selected_tool(&self) -> &str {
        self.spec.name()
    }
    pub fn finished(&self) -> bool {
        self.states.iter().any(|s| matches!(s.stage, Stage::End))
    }
    pub fn accept(&mut self, id: u32) -> Result<()> {
        let atoms = self
            .vocab
            .atoms
            .get(id as usize)
            .and_then(Option::as_ref)
            .context("forbidden native control token")?;
        let next = advance(&self.states, atoms, &self.spec);
        ensure!(
            !next.is_empty(),
            "selected token violates required named tool/JSON grammar"
        );
        self.states = next;
        self.generated.push(id);
        Ok(())
    }
    pub fn allowed(&self) -> Result<AllowedNativeTokens> {
        ensure!(!self.finished(), "cannot sample after constrained EOS");
        let began = Instant::now();
        let mut bits = vec![0u32; VOCAB.div_ceil(32)];
        fn visit(cursor: &ForcedNamedCursor, node: usize, states: &[Machine], bits: &mut [u32]) {
            for &id in &cursor.vocab.nodes[node].ids {
                bits[id as usize / 32] |= 1 << (id % 32);
            }
            for (&a, &child) in &cursor.vocab.nodes[node].children {
                let next = advance(states, &[a], &cursor.spec);
                if !next.is_empty() {
                    visit(cursor, child, &next, bits);
                }
            }
        }
        visit(self, 0, &self.states, &mut bits);
        let allowed = bits.iter().map(|v| v.count_ones() as usize).sum();
        ensure!(
            allowed > 0,
            "forced tool grammar has no valid native continuation"
        );
        let bytes = bits
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        Ok(AllowedNativeTokens {
            bits,
            identity: ConstraintIdentity {
                policy: POLICY,
                named_tool: self.spec.name.clone(),
                schema_sha256: self.spec.sha256.clone(),
                request_sha256: self.request.clone(),
                tokenizer_sha256: self.vocab.sha256.clone(),
                prompt_ids_sha256: self.prompt_sha.clone(),
                generated_prefix_sha256: ids_hash(&self.generated),
                generated_prefix_tokens: self.generated.len(),
                allowed_native_tokens: allowed,
                mask_sha256: hash(&bytes),
                build_cpu_millis: began.elapsed().as_secs_f64() * 1000.,
                constrained_greedy_before_local_maxloc: true,
            },
        })
    }
}

/// Read-only metadata is carried by opaque native model/session receipts. It
/// is never accepted from JSON as authority. The actual native head constructs
/// it only after masked kernel status and global selected-ID membership pass.
#[derive(Clone, Debug, Serialize)]
pub struct ConstraintExecution {
    pub mask: ConstraintIdentity,
    pub sampler_source_id_hex: String,
    pub sampler_ptx_sha256: String,
    pub selected_native_id: u32,
    pub local_mask_kernel_status_checked: bool,
    pub global_selected_allowed: bool,
}
fn hex(s: &str, n: usize) -> bool {
    s.len() == n && s.bytes().all(|b| b.is_ascii_hexdigit())
}
/// CPU association check used by the trusted session AFTER consuming an opaque
/// CheckedModelChunk. This public metadata helper itself creates no proof.
pub fn validate_execution_chain(
    previous: Option<&ConstraintExecution>,
    first_output: bool,
    current: Option<&ConstraintExecution>,
    request: &str,
    prompt: &[u32],
    generated_before: &[u32],
    selected: u32,
) -> Result<()> {
    if let Some(c) = current {
        ensure!(
            first_output || previous.is_some(),
            "constraint cannot appear after unmasked generated heads"
        );
        let m = &c.mask;
        ensure!(m.policy==POLICY&&safe_name(&m.named_tool)&&hex(&m.schema_sha256,64)&&hex(&m.mask_sha256,64)&&hex(&m.tokenizer_sha256,64)&&hex(&c.sampler_source_id_hex,16)&&hex(&c.sampler_ptx_sha256,64)
            &&m.request_sha256==request&&m.prompt_ids_sha256==ids_hash(prompt)&&m.generated_prefix_sha256==ids_hash(generated_before)&&m.generated_prefix_tokens==generated_before.len()
            &&m.allowed_native_tokens>0&&m.allowed_native_tokens<=VOCAB&&m.build_cpu_millis.is_finite()&&m.build_cpu_millis>=0.
            &&m.constrained_greedy_before_local_maxloc&&c.selected_native_id==selected&&c.local_mask_kernel_status_checked&&c.global_selected_allowed,
            "opaque constrained head metadata is not bound to actual native prefix/kernel/selected ID");
        if let Some(p) = previous {
            ensure!(
                p.mask.policy == m.policy
                    && p.mask.named_tool == m.named_tool
                    && p.mask.schema_sha256 == m.schema_sha256
                    && p.mask.tokenizer_sha256 == m.tokenizer_sha256
                    && p.sampler_source_id_hex == c.sampler_source_id_hex
                    && p.sampler_ptx_sha256 == c.sampler_ptx_sha256,
                "constrained sampler policy/source/schema changed within request"
            );
        }
    } else {
        ensure!(
            previous.is_none(),
            "masked generated heads cannot become unconstrained"
        );
    }
    Ok(())
}

#[cfg(test)]
mod forced_tests {
    use crate::dsv41::{native_api_contract::*, tool_constraint::*};
    use serde_json::{json, Value};
    fn bridge() -> CpuBridge {
        CpuBridge::from_verified_tokenizer(&std::fs::read("test-data/tokenizer.json").unwrap()).unwrap()
    }
    fn requests() -> Vec<Value> {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/dsv41_forced_requests.json"
        ))
        .unwrap()
    }
    fn limits() -> PlanningLimits {
        PlanningLimits {
            context_tokens: 8192,
            prompt_tokens: 8192,
            output_tokens: 4096,
            atomic_requests: 16,
            request_bytes: 1024 * 1024,
        }
    }
    fn ids(text: &str) -> Vec<u32> {
        let t=tokenizers_recipe::Tokenizer::from_file("test-data/tokenizer.json").unwrap();
        t.encode(text, false).unwrap().get_ids().to_vec()
    }
    fn output(name: &str, arg: &str) -> String {
        format!("<｜DSML｜ invoke name=\"{name}\">\n<｜DSML｜ parameter name=\"value\" string=\"false\">{arg}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>")
    }
    #[test]
    fn unchanged_01_and_16_preflight_real_rust_bridge_and_all_native_token_masks() {
        let b = bridge();
        let r = requests();
        let a = b.prepare_single(r[0].clone(), limits()).unwrap();
        assert!(a.forced_named_cursor().unwrap().is_none());
        let p = b.prepare_single(r[1].clone(), limits()).unwrap();
        assert_eq!(p.semantics().output_allowance, 512);
        assert_eq!(p.semantics().thinking, false);
        assert!(p.semantics().tool_choice.contains("echo"));
        let mut c = p.forced_named_cursor().unwrap().unwrap();
        let mut generated = ids(&output(
            "echo",
            "{\"balance_delta\":7,\"operation_id\":\"op_47\"}",
        ));
        generated.push(EOS);
        let mut total = 0.;
        for (n, &id) in generated.iter().enumerate() {
            let mask = c.allowed().unwrap();
            total += mask.identity().build_cpu_millis;
            assert!(mask.allows(id), "native{id} at{n}");
            assert_eq!(mask.allows(EOS), n + 1 == generated.len());
            mask.check_binding(
                &p.semantics().original_request_sha256,
                p.prompt_ids(),
                c.generated_ids(),
            )
            .unwrap();
            c.accept(id).unwrap();
        }
        assert!(c.finished());
        assert_eq!(c.generated_ids(), generated);
        println!(
            "FORCED_PREFLIGHT prompt_tokens={} output_tokens={} mask_build_cpu_millis={total}",
            p.prompt_ids().len(),
            generated.len()
        );
    }
    #[tokio::test]
    async fn real_official_parser_requires_named_arguments_and_tool_finish() {
        let b = bridge();
        let r = requests();
        let mut generated = ids(&output(
            "echo",
            "{\"balance_delta\":7,\"operation_id\":\"op_47\"}",
        ));
        generated.push(EOS);
        let t = CpuTranscript {
            token_chunks: generated.chunks(1).map(|x| x.to_vec()).collect(),
            terminal: CpuTerminal::NativeEos,
            cached_prompt_tokens: 0,
        };
        let p = b.prepare_single(r[1].clone(), limits()).unwrap();
        let v = b
            .parse_cpu_transcript(p, "forced-cpu".into(), &t)
            .await
            .unwrap();
        assert!(v.passed, "{:?}", v.error);
        let x = v.response.unwrap();
        assert_eq!(x["choices"][0]["finish_reason"], "tool_calls");
        let calls = x["choices"][0]["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "echo");
        assert_eq!(
            serde_json::from_str::<Value>(calls[0]["function"]["arguments"].as_str().unwrap())
                .unwrap(),
            json!({"value":{"balance_delta":7,"operation_id":"op_47"}})
        );
        for text in [
            output("wrong", "{}"),
            output("echo", "{bad}"),
            "</｜DSML｜ calls>".into(),
            output("echo", "{}") + "garbage",
        ] {
            let mut g = ids(&text);
            g.push(EOS);
            let p = b.prepare_single(r[1].clone(), limits()).unwrap();
            let v = b
                .parse_cpu_transcript(
                    p,
                    "forced-negative".into(),
                    &CpuTranscript {
                        token_chunks: vec![g],
                        terminal: CpuTerminal::NativeEos,
                        cached_prompt_tokens: 0,
                    },
                )
                .await
                .unwrap();
            assert!(!v.passed, "invalid constrained response passed");
        }
    }
    #[test]
    fn binding_controls_and_unicode_json_prefixes() {
        let b = bridge();
        let r = requests();
        let p = b.prepare_single(r[1].clone(), limits()).unwrap();
        let mut c = p.forced_named_cursor().unwrap().unwrap();
        let m = c.allowed().unwrap();
        assert!(!m.allows(EOS));
        assert!(m
            .check_binding(&"0".repeat(64), p.prompt_ids(), &[])
            .is_err());
        assert!(m
            .check_binding(&p.semantics().original_request_sha256, &[5], &[])
            .is_err());
        assert!(m
            .check_binding(&p.semantics().original_request_sha256, p.prompt_ids(), &[5])
            .is_err());
        let mut g = ids(&output(
            "echo",
            "{\"unicode\":\"東京🌍\",\"null\":null,\"nested\":[true,12.5,-3e-2]}",
        ));
        g.push(EOS);
        for id in g {
            c.accept(id).unwrap();
        }
        assert!(c.finished());
        assert!(c.accept(EOS).is_err());
        for change in ["missing", "required"] {
            let mut v = r[1].clone();
            v["tool_choice"] = if change == "missing" {
                json!({"type":"function","function":{"name":"missing"}})
            } else {
                json!("required")
            };
            assert!(b.prepare_single(v, limits()).is_err());
        }
    }
}
#[cfg(test)]
mod metadata_chain_tests {
    use crate::dsv41::tool_constraint::*;
    fn execution() -> ConstraintExecution {
        ConstraintExecution {
            mask: ConstraintIdentity {
                policy: POLICY,
                named_tool: "echo".into(),
                schema_sha256: "1".repeat(64),
                request_sha256: "2".repeat(64),
                tokenizer_sha256: "3".repeat(64),
                prompt_ids_sha256: ids_hash(&[5, 6]),
                generated_prefix_sha256: ids_hash(&[]),
                generated_prefix_tokens: 0,
                allowed_native_tokens: 4,
                mask_sha256: "4".repeat(64),
                build_cpu_millis: 1.,
                constrained_greedy_before_local_maxloc: true,
            },
            sampler_source_id_hex: "5".repeat(16),
            sampler_ptx_sha256: "6".repeat(64),
            selected_native_id: 42,
            local_mask_kernel_status_checked: true,
            global_selected_allowed: true,
        }
    }
    #[test]
    fn metadata_chain_never_creates_authority_and_refuses_drift_or_unmasked_prefix() {
        let p = execution();
        let q = &"2".repeat(64);
        validate_execution_chain(None, true, Some(&p), q, &[5, 6], &[], 42).unwrap();
        assert!(validate_execution_chain(None, false, Some(&p), q, &[5, 6], &[], 42).is_err());
        assert!(validate_execution_chain(Some(&p), false, None, q, &[5, 6], &[42], 43).is_err());
        let mut c = p.clone();
        c.mask.generated_prefix_sha256 = ids_hash(&[42]);
        c.mask.generated_prefix_tokens = 1;
        c.selected_native_id = 43;
        validate_execution_chain(Some(&p), false, Some(&c), q, &[5, 6], &[42], 43).unwrap();
        for change in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] {
            let mut b = c.clone();
            match change {
                0 => b.mask.named_tool = "other".into(),
                1 => b.mask.schema_sha256 = "0".repeat(64),
                2 => b.mask.request_sha256 = "0".repeat(64),
                3 => b.mask.prompt_ids_sha256 = "0".repeat(64),
                4 => b.mask.generated_prefix_sha256 = "0".repeat(64),
                5 => b.mask.generated_prefix_tokens = 0,
                6 => b.mask.allowed_native_tokens = 0,
                7 => b.global_selected_allowed = false,
                8 => b.local_mask_kernel_status_checked = false,
                9 => b.selected_native_id = 1,
                10 => b.sampler_ptx_sha256 = "0".repeat(64),
                _ => b.mask.constrained_greedy_before_local_maxloc = false,
            };
            assert!(
                validate_execution_chain(Some(&p), false, Some(&b), q, &[5, 6], &[42], 43).is_err(),
                "case{change}"
            );
        }
    }
}

#[cfg(test)]
mod typed_mode_tests {
    use super::*;
    use crate::dsv41::native_api_contract::{CpuBridge, PlanningLimits};
    fn ids(s: &str) -> Vec<u32> {
        let t=tokenizers_recipe::Tokenizer::from_file("test-data/tokenizer.json").unwrap();
        t.encode(s, false).unwrap().get_ids().to_vec()
    }
    #[test]
    fn integer_property_never_admits_string_mode_or_a_string_json_root() {
        let bytes=std::fs::read("test-data/tokenizer.json").unwrap();
        let b = CpuBridge::from_verified_tokenizer(&bytes).unwrap();
        let requests: Vec<Value> = serde_json::from_str(include_str!(
            "../../tests/fixtures/dsv41_forced_requests.json"
        ))
        .unwrap();
        let mut r = requests[1].clone();
        r["tools"][0]["function"]["parameters"]["properties"]["value"] =
            serde_json::json!({"type":"integer"});
        let limits = PlanningLimits {
            context_tokens: 8192,
            prompt_tokens: 8192,
            output_tokens: 4096,
            atomic_requests: 16,
            request_bytes: 1048576,
        };
        let p = b.prepare_single(r.clone(), limits).unwrap();
        let prefix = "<｜DSML｜ invoke name=\"echo\"><｜DSML｜ parameter name=\"value\" string=\"";
        let mut c = p.forced_named_cursor().unwrap().unwrap();
        for id in ids(prefix) {
            c.accept(id).unwrap();
        }
        let m = c.allowed().unwrap();
        let true_ids = ids("true");
        assert!(true_ids.len() == 1);
        assert!(!m.allows(true_ids[0]));
        assert!(c.accept(true_ids[0]).is_err());
        let mut c = p.forced_named_cursor().unwrap().unwrap();
        for id in ids(&(prefix.to_owned() + "false\">")) {
            c.accept(id).unwrap();
        }
        assert!(ids("\"wrong\"").iter().any(|&id| c.accept(id).is_err()));
        let mut c = p.forced_named_cursor().unwrap().unwrap();
        for id in ids(&(prefix.to_owned()
            + "false\">7</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"))
        {
            c.accept(id).unwrap();
        }
        c.accept(EOS).unwrap();
        assert!(c.finished());
        let json_start = prefix.to_owned() + "false\">";
        let mut bad = p.forced_named_cursor().unwrap().unwrap();
        for id in ids(&(json_start.clone() + "1.5")) {
            bad.accept(id).unwrap();
        }
        let whitespace = ids(" ");
        assert_eq!(whitespace.len(), 1);
        assert!(!bad.allowed().unwrap().allows(whitespace[0]));
        assert!(bad.accept(whitespace[0]).is_err());
        let mut good = p.forced_named_cursor().unwrap().unwrap();
        for id in
            ids(&(json_start + "1.5e1 </｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"))
        {
            good.accept(id).unwrap();
        }
        good.accept(EOS).unwrap();
        assert!(good.finished());
        for schema in [
            serde_json::json!({"enum":[7]}),
            serde_json::json!({"type":"object","properties":{"x":{}}}),
            serde_json::json!({"type":"integer","minimum":1}),
        ] {
            let mut q = r.clone();
            q["tools"][0]["function"]["parameters"]["properties"]["value"] = schema;
            assert!(
                b.prepare_single(q, limits).is_err(),
                "unimplemented schema-prefix policy was silently admitted"
            );
        }
    }
}
