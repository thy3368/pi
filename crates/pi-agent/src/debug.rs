use pi_ai::Context;

use crate::types::AgentConfig;

const DEBUG_PROVIDER_REQUEST_ENV: &str = "PI_AI_DEBUG_PROVIDER_REQUEST";

pub(crate) fn log_llm_context(turn: u32, config: &AgentConfig, context: &Context) {
    if std::env::var(DEBUG_PROVIDER_REQUEST_ENV).as_deref() != Ok("2") {
        return;
    }

    eprintln!("{}", format_llm_context(turn, config, context));
}

fn format_llm_context(turn: u32, config: &AgentConfig, context: &Context) -> String {
    let system_prompt = context.system_prompt.as_deref().unwrap_or("");
    let messages = serde_json::to_string_pretty(&context.messages)
        .unwrap_or_else(|_| format!("{:?}", context.messages));
    let tools = serde_json::to_string_pretty(&context.tools)
        .unwrap_or_else(|_| format!("{:?}", context.tools));

    format!(
        "[pi-agent llm context] turn={turn}\nprovider: {}\napi: {}\nmodel: {}\nsystem_prompt:\n{}\nmessages:\n{}\ntools:\n{}",
        config.model.provider, config.model.api, config.model.id, system_prompt, messages, tools
    )
}

#[cfg(test)]
mod tests {
    use pi_ai::{Content, Message, Model, Tool, ToolResultMessage};
    use serde_json::json;

    use super::*;

    #[test]
    fn formats_agent_context_without_provider_wire_details_or_credentials() {
        let mut config = AgentConfig::new(
            Model::openai_compat(
                "vokotoken",
                "gpt-5.5",
                "https://vokotoken.cc/v1",
                200_000,
                8_192,
            ),
            "system prompt",
        );
        config.stream_options.api_key = Some("secret-key".to_string());

        let context = Context {
            system_prompt: Some("system prompt".to_string()),
            messages: vec![
                Message::user_text("hello"),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "call_1".to_string(),
                    tool_name: "web_fetch".to_string(),
                    content: vec![Content::text("tool output")],
                    is_error: false,
                    timestamp: 1,
                }),
            ],
            tools: vec![Tool {
                name: "web_fetch".to_string(),
                description: "fetch a URL".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "url": { "type": "string" }
                    }
                }),
            }],
        };

        let output = format_llm_context(2, &config, &context);

        assert!(output.starts_with("[pi-agent llm context] turn=2"));
        assert!(output.contains("provider: vokotoken"));
        assert!(output.contains("api: openai-completions"));
        assert!(output.contains("model: gpt-5.5"));
        assert!(output.contains("system prompt"));
        assert!(output.contains("\"role\": \"user\""));
        assert!(output.contains("\"role\": \"toolResult\""));
        assert!(output.contains("\"name\": \"web_fetch\""));
        assert!(output.contains("\"tool_name\": \"web_fetch\""));
        assert!(!output.contains("secret-key"));
        assert!(!output.contains("api_key"));
        assert!(!output.contains("https://vokotoken.cc/v1"));
        assert!(!output.contains("Authorization"));
        assert!(!output.contains("headers:"));
        assert!(!output.contains("body:"));
    }
}
