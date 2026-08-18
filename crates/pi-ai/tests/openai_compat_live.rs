//! Live integration test for Vokotoken's OpenAI-compatible Chat Completions API.
//!
//! This test is intentionally not ignored: it verifies the default test suite
//! can exercise a real OpenAI-compatible streaming endpoint when credentials
//! are configured.

use futures::StreamExt;
use pi_ai::{
    stream_simple, AssistantMessageEvent, Content, Context, Message, Model, StreamOptions,
};

fn api_key() -> String {
    std::env::var("PI_AI_VOKOTOKEN_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .expect("set PI_AI_VOKOTOKEN_API_KEY or OPENAI_API_KEY to run the Vokotoken live test")
}

#[tokio::test]
async fn vokotoken_openai_compat_streams_chat_completions() {
    let model = Model::openai_compat(
        "vokotoken",
        "gpt-5.5",
        "https://vokotoken.cc/v1",
        200_000,
        8_192,
    );
    let context = Context {
        system_prompt: None,
        messages: vec![Message::user_text("今天最热的新闻是什么？")],
        tools: Vec::new(),
    };
    let options = StreamOptions {
        api_key: Some(api_key()),
        max_tokens: Some(512),
        temperature: Some(0.0),
        ..Default::default()
    };

    let mut stream = stream_simple(&model, &context, &options)
        .await
        .expect("stream_simple should create a Vokotoken OpenAI-compatible stream");

    let mut saw_start = false;
    let mut saw_text = false;
    let mut saw_done = false;

    while let Some(event) = stream.next().await {
        match event.expect("Vokotoken stream event should be valid") {
            AssistantMessageEvent::Start => {
                saw_start = true;
            }
            AssistantMessageEvent::TextDelta { delta, .. } => {
                if !delta.trim().is_empty() {
                    saw_text = true;
                }
            }
            AssistantMessageEvent::TextEnd { content, .. } => {
                if !content.trim().is_empty() {
                    saw_text = true;
                }
            }
            AssistantMessageEvent::Done { message, .. } => {
                saw_done = true;
                println!("Vokotoken returned message: {message:?}");
                saw_text |= message.content.iter().any(
                    |content| matches!(content, Content::Text { text } if !text.trim().is_empty()),
                );

                assert_eq!(message.provider, "vokotoken");
                assert_eq!(message.api, "openai-completions");
                assert!(
                    message.error_message.is_none(),
                    "expected no provider error message, got {:?}",
                    message.error_message
                );
            }
            AssistantMessageEvent::Error { error, .. } => {
                panic!(
                    "Vokotoken stream returned an error event: {:?}",
                    error.error_message
                );
            }
            _ => {}
        }
    }

    assert!(saw_start, "expected a Start event");
    assert!(
        saw_text,
        "expected a non-empty text delta or final text content"
    );
    assert!(saw_done, "expected a Done event");
}
