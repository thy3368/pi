//! Live integration test for Vokotoken's OpenAI-compatible Chat Completions API
//! through the pi-agent tool-calling loop.
//!
//! This test is intentionally not ignored: it verifies a real model can request
//! the built-in `web_fetch` tool, receive the tool result, and produce a final
//! answer.

use std::sync::Arc;

use pi_agent::{run_agent, AgentConfig};
use pi_ai::{Content, Message, Model, StreamOptions};

fn api_key() -> String {
    std::env::var("PI_AI_VOKOTOKEN_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .expect("set PI_AI_VOKOTOKEN_API_KEY or OPENAI_API_KEY to run the Vokotoken live test")
}

#[tokio::test]
async fn vokotoken_openai_compat_runs_web_fetch_tool_chain() {
    let model = Model::openai_compat(
        "vokotoken",
        "gpt-5.5",
        "https://vokotoken.cc/v1",
        200_000,
        8_192,
    );

    let mut config = AgentConfig::new(
        model,
        concat!(
            "You are testing an agent tool-calling loop. ",
            "Before answering the user, call the web_fetch tool exactly once with ",
            "url=https://news.baidu.com/ and max_chars=4000. ",
            "After the tool result is returned, answer the user's question briefly in Chinese. ",
            "Do not answer before calling web_fetch."
        ),
    )
    .with_tools(vec![Arc::new(pi_agent::tools::web_fetch::WebFetchTool)])
    .with_max_turns(3);
    config.stream_options = StreamOptions {
        api_key: Some(api_key()),
        max_tokens: Some(1024),
        temperature: Some(0.0),
        ..Default::default()
    };

    let run = run_agent(&config, Message::user_text("今天最热的新闻是什么？"), None)
        .await
        .expect("run_agent should complete a Vokotoken OpenAI-compatible tool-chain run");

    println!("Vokotoken pi-agent transcript:");
    for (idx, message) in run.messages.iter().enumerate() {
        match message {
            Message::Assistant(message) => println!("message[{idx}] assistant: {message:?}"),
            Message::ToolResult(result) => println!("message[{idx}] tool_result: {result:?}"),
            Message::User { .. } => println!("message[{idx}] user: {message:?}"),
        }
    }

    assert!(
        !run.stopped_at_turn_limit,
        "agent should finish after the tool result and final assistant answer"
    );

    let saw_web_fetch_call = run.messages.iter().any(|message| {
        matches!(
            message,
            Message::Assistant(assistant)
                if assistant.content.iter().any(|content| {
                    matches!(content, Content::ToolCall { name, .. } if name == "web_fetch")
                })
        )
    });
    assert!(
        saw_web_fetch_call,
        "expected an assistant tool call named web_fetch"
    );

    let web_fetch_result_text = run.messages.iter().find_map(|message| match message {
        Message::ToolResult(result) if result.tool_name == "web_fetch" && !result.is_error => Some(
            result
                .content
                .iter()
                .filter_map(Content::as_text)
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        _ => None,
    });
    let web_fetch_result_text =
        web_fetch_result_text.expect("expected a successful web_fetch tool result");
    assert!(
        web_fetch_result_text.contains("GET https://news.baidu.com/ ["),
        "expected web_fetch result to include the fetched URL and status, got:\n{web_fetch_result_text}"
    );

    let final_assistant = run
        .messages
        .iter()
        .rev()
        .find_map(|message| match message {
            Message::Assistant(assistant) => Some(assistant),
            _ => None,
        })
        .expect("expected at least one assistant message");
    assert!(
        final_assistant
            .content
            .iter()
            .filter_map(Content::as_text)
            .any(|text| !text.trim().is_empty()),
        "expected the final assistant message to contain non-empty text"
    );

    for message in &run.messages {
        let Message::Assistant(assistant) = message else {
            continue;
        };
        assert_eq!(assistant.provider, "vokotoken");
        assert_eq!(assistant.api, "openai-completions");
        assert!(
            assistant.error_message.is_none(),
            "expected no provider error message, got {:?}",
            assistant.error_message
        );
    }
}
