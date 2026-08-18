//! OpenAI Responses API provider — streaming via Server-Sent Events.
//!
//! Targets the `/responses` endpoint used by OpenAI's reasoning models such as
//! the `o-series` and `gpt-5*` line, which differs from the Chat Completions
//! shape and emits typed `response.*` SSE events.

use std::collections::BTreeMap;

use async_stream::stream;
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::providers::debug::log_provider_request;
use crate::providers::Provider;
use crate::retry::{classify_status, parse_retry_after, with_retry, Attempt, RetryConfig};
use crate::stream::AssistantMessageEventStream;
use crate::types::{
    now_ms, AssistantMessage, AssistantMessageEvent, Content, Context, Message, Model, StopReason,
    StreamOptions, ThinkingLevel, Usage,
};

pub struct OpenAiResponsesProvider {
    client: reqwest::Client,
}

impl OpenAiResponsesProvider {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for OpenAiResponsesProvider {
    fn default() -> Self {
        Self::new()
    }
}

fn reasoning_effort(level: ThinkingLevel) -> Option<&'static str> {
    match level {
        ThinkingLevel::Off => None,
        ThinkingLevel::Minimal | ThinkingLevel::Low => Some("low"),
        ThinkingLevel::Medium => Some("medium"),
        ThinkingLevel::High | ThinkingLevel::Xhigh => Some("high"),
    }
}

fn convert_input(messages: &[Message]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    for m in messages {
        match m {
            Message::User { content, .. } => {
                let text = content
                    .iter()
                    .filter_map(|c| c.as_text().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
                    .join("");
                out.push(json!({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                }));
            }
            Message::Assistant(a) => {
                let mut text = String::new();
                let mut tool_calls: Vec<Value> = Vec::new();
                for c in &a.content {
                    match c {
                        Content::Text { text: t } => text.push_str(t),
                        Content::ToolCall {
                            id,
                            name,
                            arguments,
                        } => {
                            tool_calls.push(json!({
                                "type": "function_call",
                                "call_id": id,
                                "name": name,
                                "arguments": arguments.to_string(),
                            }));
                        }
                        _ => {}
                    }
                }
                if !text.is_empty() {
                    out.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }));
                }
                for tc in tool_calls {
                    out.push(tc);
                }
            }
            Message::ToolResult(tr) => {
                let text = tr
                    .content
                    .iter()
                    .filter_map(|c| c.as_text().map(|s| s.to_string()))
                    .collect::<Vec<_>>()
                    .join("");
                out.push(json!({
                    "type": "function_call_output",
                    "call_id": tr.tool_call_id,
                    "output": text,
                }));
            }
        }
    }
    out
}

fn build_body(model: &Model, context: &Context, options: &StreamOptions) -> Value {
    let mut body = json!({
        "model": model.id,
        "input": convert_input(&context.messages),
        "stream": true,
    });
    if let Some(sp) = &context.system_prompt {
        body["instructions"] = json!(sp);
    }
    if let Some(m) = options.max_tokens {
        body["max_output_tokens"] = json!(m);
    }
    if let Some(t) = options.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(level) = options.reasoning {
        if let Some(effort) = reasoning_effort(level) {
            body["reasoning"] = json!({"effort": effort});
        }
    }
    if !context.tools.is_empty() {
        let tools: Vec<Value> = context
            .tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })
            })
            .collect();
        body["tools"] = json!(tools);
    }
    body
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    args: String,
}

#[derive(Deserialize, Debug, Default)]
struct OutputItemAdded {
    #[serde(default)]
    item: Option<OutputItem>,
}

#[derive(Deserialize, Debug, Default)]
struct OutputItem {
    #[serde(rename = "type", default)]
    item_type: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct DeltaEnvelope {
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    item_id: Option<String>,
    #[serde(default)]
    output_index: Option<usize>,
}

#[async_trait]
impl Provider for OpenAiResponsesProvider {
    async fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> Result<AssistantMessageEventStream> {
        let api_key = options
            .api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| Error::MissingApiKey("openai".into()))?;
        let base_url = options
            .base_url
            .clone()
            .unwrap_or_else(|| model.base_url.clone());
        let url = format!("{}/responses", base_url.trim_end_matches('/'));
        let body = build_body(model, context, options);
        let cancel = options.cancel.clone();
        let extra_headers: BTreeMap<String, String> = options.headers.clone();
        let mut debug_headers = BTreeMap::from([
            ("authorization".to_string(), format!("Bearer {api_key}")),
            ("accept".to_string(), "text/event-stream".to_string()),
            ("content-type".to_string(), "application/json".to_string()),
        ]);
        debug_headers.extend(extra_headers.clone());
        log_provider_request(
            &model.provider,
            &model.api,
            &model.id,
            "POST",
            &url,
            &debug_headers,
            &body,
        );

        let resp = with_retry(&RetryConfig::default(), cancel.as_ref(), |_| {
            let client = self.client.clone();
            let url = url.clone();
            let api_key = api_key.clone();
            let body = body.clone();
            let extra_headers = extra_headers.clone();
            async move {
                let mut req = client
                    .post(&url)
                    .bearer_auth(&api_key)
                    .header("accept", "text/event-stream")
                    .header("content-type", "application/json");
                for (k, v) in extra_headers {
                    req = req.header(k, v);
                }
                let r = match req.json(&body).send().await {
                    Ok(r) => r,
                    Err(e) => {
                        return if e.is_timeout() || e.is_connect() {
                            Attempt::Retry {
                                error: Error::Http(e),
                                retry_after: None,
                            }
                        } else {
                            Attempt::Fatal(Error::Http(e))
                        }
                    }
                };
                let status = r.status();
                if status.is_success() {
                    return Attempt::Ok(r);
                }
                let retry_after = r
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(parse_retry_after);
                let body = r.text().await.unwrap_or_default();
                let err = Error::ProviderError {
                    status: status.as_u16(),
                    body,
                };
                match classify_status(status.as_u16()) {
                    Some(_) => Attempt::Retry {
                        error: err,
                        retry_after,
                    },
                    None => Attempt::Fatal(err),
                }
            }
        })
        .await?;

        let api = model.api.clone();
        let provider = model.provider.clone();
        let model_id = model.id.clone();
        let cancel_for_stream = cancel.clone();

        let s = stream! {
            yield Ok(AssistantMessageEvent::Start);

            let mut sse = resp.bytes_stream().eventsource();

            let mut text_buf = String::new();
            let mut text_started = false;
            let mut text_index: usize = 0;
            // Map response output_index → local accumulator index.
            let mut item_index_map: std::collections::BTreeMap<String, usize> = Default::default();
            let mut tool_calls: Vec<PartialToolCall> = Vec::new();
            let mut tool_started: std::collections::BTreeSet<usize> = Default::default();
            let mut stop = StopReason::Stop;
            let usage = Usage::default();

            while let Some(ev) = sse.next().await {
                if let Some(c) = &cancel_for_stream {
                    if c.is_cancelled() {
                        yield Err(Error::Cancelled);
                        return;
                    }
                }
                let ev = match ev {
                    Ok(e) => e,
                    Err(e) => {
                        yield Err(Error::InvalidResponse(format!("sse: {e}")));
                        return;
                    }
                };
                if ev.data.is_empty() {
                    continue;
                }
                match ev.event.as_str() {
                    "response.output_text.delta" => {
                        let env: DeltaEnvelope = match serde_json::from_str(&ev.data) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        if let Some(d) = env.delta {
                            if !d.is_empty() {
                                if !text_started {
                                    text_started = true;
                                    yield Ok(AssistantMessageEvent::TextStart { content_index: text_index });
                                }
                                text_buf.push_str(&d);
                                yield Ok(AssistantMessageEvent::TextDelta {
                                    content_index: text_index,
                                    delta: d,
                                });
                            }
                        }
                    }
                    "response.output_item.added" => {
                        let added: OutputItemAdded = match serde_json::from_str(&ev.data) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        let Some(item) = added.item else { continue };
                        if item.item_type != "function_call" {
                            continue;
                        }
                        let id = item.call_id.or(item.id.clone()).unwrap_or_default();
                        let name = item.name.unwrap_or_default();
                        let idx = tool_calls.len();
                        tool_calls.push(PartialToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            args: String::new(),
                        });
                        if let Some(item_id) = item.id {
                            item_index_map.insert(item_id, idx);
                        }
                        let block_index = text_index
                            + if text_started { 1 } else { 0 }
                            + idx;
                        tool_started.insert(idx);
                        yield Ok(AssistantMessageEvent::ToolCallStart {
                            content_index: block_index,
                            id,
                            name,
                        });
                    }
                    "response.function_call_arguments.delta" => {
                        let env: DeltaEnvelope = match serde_json::from_str(&ev.data) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        let Some(d) = env.delta else { continue };
                        let idx = env
                            .item_id
                            .as_ref()
                            .and_then(|id| item_index_map.get(id).copied())
                            .or(env.output_index)
                            .unwrap_or_else(|| tool_calls.len().saturating_sub(1));
                        if let Some(entry) = tool_calls.get_mut(idx) {
                            entry.args.push_str(&d);
                            let block_index = text_index
                                + if text_started { 1 } else { 0 }
                                + idx;
                            yield Ok(AssistantMessageEvent::ToolCallDelta {
                                content_index: block_index,
                                delta: d,
                            });
                        }
                    }
                    "response.completed" => {
                        break;
                    }
                    _ => {}
                }
            }

            if text_started {
                yield Ok(AssistantMessageEvent::TextEnd {
                    content_index: text_index,
                    content: text_buf.clone(),
                });
                text_index += 1;
            }

            if !tool_calls.is_empty() {
                stop = StopReason::ToolUse;
            }

            let mut out_content: Vec<Content> = Vec::new();
            if text_started {
                out_content.push(Content::Text { text: text_buf.clone() });
            }
            for (i, tc) in tool_calls.into_iter().enumerate() {
                let args: Value = if tc.args.is_empty() {
                    Value::Object(Default::default())
                } else {
                    serde_json::from_str(&tc.args).unwrap_or(Value::Object(Default::default()))
                };
                let block_index = text_index + i;
                yield Ok(AssistantMessageEvent::ToolCallEnd {
                    content_index: block_index,
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments: args.clone(),
                });
                out_content.push(Content::ToolCall {
                    id: tc.id,
                    name: tc.name,
                    arguments: args,
                });
            }

            let message = AssistantMessage {
                content: out_content,
                api,
                provider,
                model: model_id,
                usage,
                stop_reason: stop,
                error_message: None,
                timestamp: now_ms(),
            };
            yield Ok(AssistantMessageEvent::Done { reason: stop, message });
        };

        Ok(s.boxed())
    }
}
