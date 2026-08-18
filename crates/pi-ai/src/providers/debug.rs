use std::collections::BTreeMap;

use serde_json::Value;
pub(crate) const DEBUG_PROVIDER_REQUEST_ENV: &str = "PI_AI_DEBUG_PROVIDER_REQUEST";

pub(crate) fn log_provider_request(
    provider: &str,
    api: &str,
    model: &str,
    method: &str,
    url: &str,
    headers: &BTreeMap<String, String>,
    body: &Value,
) {
    if std::env::var(DEBUG_PROVIDER_REQUEST_ENV).as_deref() != Ok("1") {
        return;
    }

    eprintln!(
        "{}",
        format_provider_request(provider, api, model, method, url, headers, body)
    );
}

pub(crate) fn format_provider_request(
    provider: &str,
    api: &str,
    model: &str,
    method: &str,
    url: &str,
    headers: &BTreeMap<String, String>,
    body: &Value,
) -> String {
    let redacted_headers = headers
        .iter()
        .map(|(name, value)| {
            let value = if is_sensitive_header(name) {
                "<redacted>".to_string()
            } else {
                value.clone()
            };
            format!("{name}: {value}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let body = serde_json::to_string_pretty(body).unwrap_or_else(|_| body.to_string());

    format!(
        "[pi-ai provider request]\nprovider: {provider}\napi: {api}\nmodel: {model}\nmethod: {method}\nurl: {}\nheaders:\n{}\nbody:\n{}",
        redact_url(url),
        redacted_headers,
        body
    )
}

fn is_sensitive_header(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name == "authorization"
        || name == "proxy-authorization"
        || name == "x-api-key"
        || name == "api-key"
        || name.ends_with("-api-key")
}

pub(crate) fn redact_url(url: &str) -> String {
    let Some((prefix, suffix)) = url.split_once('?') else {
        return url.to_string();
    };
    let (query, fragment) = suffix
        .split_once('#')
        .map_or((suffix, None), |(query, fragment)| (query, Some(fragment)));

    let query = query
        .split('&')
        .map(|pair| {
            let Some((key, _value)) = pair.split_once('=') else {
                return pair.to_string();
            };
            if key.eq_ignore_ascii_case("key") {
                format!("{key}=<redacted>")
            } else {
                pair.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("&");

    match fragment {
        Some(fragment) => format!("{prefix}?{query}#{fragment}"),
        None => format!("{prefix}?{query}"),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn redacts_sensitive_headers() {
        let headers = BTreeMap::from([
            ("Authorization".to_string(), "Bearer secret".to_string()),
            ("accept".to_string(), "text/event-stream".to_string()),
            ("x-api-key".to_string(), "secret".to_string()),
        ]);

        let output = format_provider_request(
            "test-provider",
            "test-api",
            "test-model",
            "POST",
            "https://example.test/v1/messages",
            &headers,
            &json!({"messages": [{"role": "user", "content": "hello"}]}),
        );

        assert!(output.contains("Authorization: <redacted>"));
        assert!(output.contains("x-api-key: <redacted>"));
        assert!(output.contains("accept: text/event-stream"));
        assert!(!output.contains("Bearer secret"));
        assert!(!output.contains("x-api-key: secret"));
    }

    #[test]
    fn redacts_google_key_query_param() {
        let redacted =
            redact_url("https://generativelanguage.googleapis.com/v1beta/models/gemini:streamGenerateContent?alt=sse&key=secret");

        assert_eq!(
            redacted,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini:streamGenerateContent?alt=sse&key=<redacted>"
        );
        assert!(!redacted.contains("secret"));
    }

    #[test]
    fn preserves_prompt_and_messages_in_body() {
        let output = format_provider_request(
            "openai",
            "openai-completions",
            "gpt-test",
            "POST",
            "https://example.test/chat/completions",
            &BTreeMap::new(),
            &json!({
                "messages": [{"role": "user", "content": "debug this exact prompt"}],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            }),
        );

        assert!(output.contains("debug this exact prompt"));
        assert!(output.contains("lookup"));
    }
}
