use serde::{Deserialize, Serialize};

use crate::messages::InputMessage;
use crate::tools::{ToolChoice, ToolDefinition};

/// A conversation and its prompt configuration.
///
/// Sampling parameters and transport options belong to the caller. Thinking
/// mode is enabled by default.
#[derive(Debug, Clone)]
pub struct Conversation {
    /// Messages in conversation order.
    pub messages: Vec<InputMessage>,
    /// Whether the next assistant turn uses thinking mode.
    ///
    /// Rendering uses it to select the thinking marker that opens the next
    /// assistant turn: [`ThinkingStart`] when `true` and [`ThinkingEnd`]
    /// otherwise. Rendering writes the reasoning effort prompt only when it is
    /// `true`, and V4.1 rendering includes the reasoning content of assistant
    /// messages only when it is `true`.
    ///
    /// [`ThinkingStart`]: crate::special_tokens::SpecialToken::ThinkingStart
    /// [`ThinkingEnd`]: crate::special_tokens::SpecialToken::ThinkingEnd
    pub thinking_mode: bool,
    /// Available client tools, in declaration order.
    pub tools: Vec<ToolDefinition>,
    /// Tool selection after protocol-specific normalization.
    pub tool_choice: ToolChoice,
    /// Reasoning effort for the reasoning effort prompt, which rendering
    /// writes only in thinking mode.
    ///
    /// `None` and [`ReasoningEffort::Xhigh`] render the same as
    /// [`ReasoningEffort::High`]. V4 rendering writes no reasoning effort prompt
    /// for [`ReasoningEffort::Low`].
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Requested answer format used during prompt rendering.
    pub response_format: ResponseFormat,
}

impl Default for Conversation {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            thinking_mode: true,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            reasoning_effort: None,
            response_format: ResponseFormat::Text,
        }
    }
}

/// The requested answer format.
///
/// Rendering describes `JsonObject` and `JsonSchema` in the prompt. Constrained
/// decoding, which enforces a format during inference, belongs to the caller.
#[derive(Debug, Clone, Default)]
pub enum ResponseFormat {
    #[default]
    Text,
    JsonObject,
    /// A JSON Schema the answer must follow, rendered into the prompt.
    JsonSchema(serde_json::Value),
    /// A regular expression the answer must match. Rendering does not write it
    /// into the prompt.
    Regex(String),
}

/// A validated V4.1 reasoning score. Its private field prevents renderers from
/// receiving zero or a score above100 through a constructed `Score` variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct ReasoningScore(u8);

impl ReasoningScore {
    pub fn get(self) -> u8 {
        self.0
    }
}

impl TryFrom<u8> for ReasoningScore {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        if (1..=100).contains(&value) {
            Ok(Self(value))
        } else {
            Err("reasoning score must be an integer in1..=100")
        }
    }
}

impl From<ReasoningScore> for u8 {
    fn from(value: ReasoningScore) -> Self {
        value.0
    }
}

/// Protocol-independent reasoning effort. Named values retain their upstream
/// JSON spellings; V4.1 numeric values serialize as integers, without rounding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    High,
    Xhigh,
    Max,
    Score(ReasoningScore),
}

impl Serialize for ReasoningEffort {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Low => serializer.serialize_str("low"),
            Self::High => serializer.serialize_str("high"),
            Self::Xhigh => serializer.serialize_str("xhigh"),
            Self::Max => serializer.serialize_str("max"),
            Self::Score(score) => serializer.serialize_u8(score.get()),
        }
    }
}

impl<'de> Deserialize<'de> for ReasoningEffort {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct EffortVisitor;
        impl<'de> serde::de::Visitor<'de> for EffortVisitor {
            type Value = ReasoningEffort;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("low/high/xhigh/max or an integer reasoning score in1..=100")
            }
            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                match value {
                    "low" => Ok(ReasoningEffort::Low),
                    "high" => Ok(ReasoningEffort::High),
                    "xhigh" => Ok(ReasoningEffort::Xhigh),
                    "max" => Ok(ReasoningEffort::Max),
                    _ => Err(E::custom("invalid named reasoning effort")),
                }
            }
            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                let byte = u8::try_from(value).map_err(E::custom)?;
                ReasoningScore::try_from(byte)
                    .map(ReasoningEffort::Score)
                    .map_err(E::custom)
            }
            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                self.visit_u64(u64::try_from(value).map_err(E::custom)?)
            }
        }
        deserializer.deserialize_any(EffortVisitor)
    }
}

#[cfg(test)]
mod numeric_reasoning_tests {
    use super::*;

    #[test]
    fn all_numeric_scores_round_trip_without_mapping() {
        for value in 1..=100_u8 {
            let effort = ReasoningEffort::Score(ReasoningScore::try_from(value).unwrap());
            let encoded = serde_json::to_string(&effort).unwrap();
            assert_eq!(encoded, value.to_string());
            assert_eq!(
                serde_json::from_str::<ReasoningEffort>(&encoded).unwrap(),
                effort
            );
        }
    }

    #[test]
    fn invalid_numbers_and_named_spellings_fail() {
        for value in [
            "0",
            "101",
            "-1",
            "256",
            "42.0",
            "true",
            "null",
            "\"42\"",
            "\"medium\"",
        ] {
            assert!(
                serde_json::from_str::<ReasoningEffort>(value).is_err(),
                "{value}"
            );
        }
        for name in ["low", "high", "xhigh", "max"] {
            let text = serde_json::to_string(name).unwrap();
            let effort = serde_json::from_str::<ReasoningEffort>(&text).unwrap();
            assert_eq!(serde_json::to_string(&effort).unwrap(), text);
        }
        assert!(ReasoningScore::try_from(0).is_err());
        assert!(ReasoningScore::try_from(101).is_err());
    }
}
