//! WebSearch 工具处理模块
//!
//! 实现 Anthropic WebSearch 请求到 Kiro MCP 的转换和响应生成

use std::convert::Infallible;

use axum::{
    body::Body,
    http::{StatusCode, header},
    response::{IntoResponse, Json, Response},
};
use bytes::Bytes;
use futures::{Stream, stream};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use super::stream::SseEvent;
use super::types::{ErrorResponse, MessagesRequest};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WebSearchCacheContext {
    pub cache_creation_input_tokens: i32,
    pub cache_read_input_tokens: i32,
    pub cache_creation_5m_input_tokens: i32,
    pub cache_creation_1h_input_tokens: i32,
}

fn billed_input_tokens(
    input_tokens: i32,
    cache_creation_input_tokens: i32,
    cache_read_input_tokens: i32,
) -> i32 {
    input_tokens
        .saturating_sub(cache_creation_input_tokens)
        .saturating_sub(cache_read_input_tokens)
        .max(0)
}

fn resolve_cache_usage(
    cache_tracker: &crate::anthropic::cache_tracker::CacheTracker,
    credential_id: u64,
    profile: &crate::anthropic::cache_tracker::CacheProfile,
) -> WebSearchCacheContext {
    let result = cache_tracker.compute(credential_id, profile);
    WebSearchCacheContext {
        cache_creation_input_tokens: result.cache_creation_input_tokens,
        cache_read_input_tokens: result.cache_read_input_tokens,
        cache_creation_5m_input_tokens: result.cache_creation_5m_input_tokens,
        cache_creation_1h_input_tokens: result.cache_creation_1h_input_tokens,
    }
}

const WEB_SEARCH_PREFIX: &str = "Perform a web search for the query: ";

/// MCP 请求
#[derive(Debug, Serialize)]
pub struct McpRequest {
    pub id: String,
    pub jsonrpc: String,
    pub method: String,
    pub params: McpParams,
}

/// MCP 请求参数
#[derive(Debug, Serialize)]
pub struct McpParams {
    pub name: String,
    pub arguments: McpArguments,
}

/// MCP 参数
#[derive(Debug, Serialize)]
pub struct McpArguments {
    pub query: String,
}

/// MCP 响应
#[derive(Debug, Deserialize)]
pub struct McpResponse {
    pub error: Option<McpError>,
    #[allow(dead_code)]
    pub id: String,
    #[allow(dead_code)]
    pub jsonrpc: String,
    pub result: Option<McpResult>,
}

/// MCP 错误
#[derive(Debug, Deserialize)]
pub struct McpError {
    pub code: Option<i32>,
    pub message: Option<String>,
}

/// MCP 结果
#[derive(Debug, Deserialize)]
pub struct McpResult {
    pub content: Vec<McpContent>,
    #[serde(rename = "isError")]
    #[allow(dead_code)]
    pub is_error: bool,
}

/// MCP 内容
#[derive(Debug, Deserialize)]
pub struct McpContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

/// WebSearch 搜索结果
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct WebSearchResults {
    pub results: Vec<WebSearchResult>,
    #[serde(rename = "totalResults")]
    pub total_results: Option<i32>,
    pub query: Option<String>,
    pub error: Option<String>,
}

/// 单个搜索结果
#[derive(Debug, Deserialize, Clone)]
#[allow(dead_code)]
pub struct WebSearchResult {
    pub title: String,
    pub url: String,
    pub snippet: Option<String>,
    #[serde(rename = "publishedDate")]
    pub published_date: Option<i64>,
    pub id: Option<String>,
    pub domain: Option<String>,
    #[serde(rename = "maxVerbatimWordLimit")]
    pub max_verbatim_word_limit: Option<i32>,
    #[serde(rename = "publicDomain")]
    pub public_domain: Option<bool>,
}

/// 检查请求是否包含 WebSearch 工具
///
/// 只要 tools 中出现 web_search（按 name 或 type 判断），就认为应走本地 WebSearch 处理。
pub fn has_web_search_tool(req: &MessagesRequest) -> bool {
    req.tools.as_ref().is_some_and(|tools| {
        tools
            .iter()
            .any(|t| t.name == "web_search" || t.is_web_search())
    })
}

fn tool_choice_requests_web_search(req: &MessagesRequest) -> bool {
    let Some(choice) = req.tool_choice.as_ref() else {
        return false;
    };

    let Some(obj) = choice.as_object() else {
        return false;
    };

    // Anthropic 常见形态：{"type":"tool","name":"web_search"}
    let name = obj
        .get("name")
        .and_then(|v| v.as_str())
        .or_else(|| obj.get("tool_name").and_then(|v| v.as_str()));

    if name != Some("web_search") {
        return false;
    }

    // 若包含 type 字段，仅当 type=tool 才视为“强制调用”
    match obj.get("type").and_then(|v| v.as_str()) {
        Some("tool") => true,
        Some(_) => false,
        None => true,
    }
}

fn is_only_web_search_tool(req: &MessagesRequest) -> bool {
    req.tools.as_ref().is_some_and(|tools| {
        tools.len() == 1
            && tools
                .first()
                .is_some_and(|t| t.name == "web_search" || t.is_web_search())
    })
}

fn extract_last_user_text(req: &MessagesRequest) -> Option<String> {
    let msg = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .or_else(|| req.messages.last())?;

    match &msg.content {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(arr) => {
            let first_block = arr.first()?;
            if first_block.get("type")?.as_str()? == "text" {
                Some(first_block.get("text")?.as_str()?.to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn request_explicit_web_search_prefix(req: &MessagesRequest) -> bool {
    extract_last_user_text(req)
        .map(|t| t.trim_start().starts_with(WEB_SEARCH_PREFIX))
        .unwrap_or(false)
}

/// 判断当前请求是否应走“本地 WebSearch”处理。
///
/// 注意：`tools` 里包含 `web_search` 仅代表“可用工具”，并不代表这次一定要执行搜索。
/// 若不加额外条件，容易把普通对话/任务指令误当成搜索查询，导致 MCP 侧返回 -32602。
/// 本地 WebSearch 不经过 Kiro 上游链路，因此不会收到 meteringEvent，也不会返回 credit_* usage 字段。
pub fn should_handle_websearch_request(req: &MessagesRequest) -> bool {
    if !has_web_search_tool(req) {
        return false;
    }

    // 1) tool_choice 强制选择 web_search
    if tool_choice_requests_web_search(req) {
        return true;
    }

    // 2) 兼容旧客户端：仅提供 web_search 单工具时，视为“纯 WebSearch 请求”
    if is_only_web_search_tool(req) {
        return true;
    }

    // 3) 兼容 Claude Code 风格前缀
    request_explicit_web_search_prefix(req)
}

/// 从请求的 tools 列表中移除 web_search 工具。
///
/// 当请求包含混合工具（web_search + 其他工具）时，剔除 web_search 后转发上游。
/// 若剔除后 tools 为空，则设为 None。
pub fn strip_web_search_tools(req: &mut MessagesRequest) {
    if let Some(tools) = req.tools.as_mut() {
        tools.retain(|t| t.name != "web_search" && !t.is_web_search());
        if tools.is_empty() {
            req.tools = None;
        }
    }
}

/// 从消息中提取搜索查询
///
/// 读取 messages 中最后一条 user 消息的第一个内容块（更符合多轮对话场景）
/// 并去除 "Perform a web search for the query: " 前缀
pub fn extract_search_query(req: &MessagesRequest) -> Option<String> {
    // 优先取最后一条 user 消息，否则回退到最后一条消息
    let msg = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .or_else(|| req.messages.last())?;

    // 提取文本内容
    let text = match &msg.content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => {
            // 获取第一个内容块
            let first_block = arr.first()?;
            if first_block.get("type")?.as_str()? == "text" {
                first_block.get("text")?.as_str()?.to_string()
            } else {
                return None;
            }
        }
        _ => return None,
    };

    // 去除前缀 "Perform a web search for the query: "
    let query = text
        .strip_prefix(WEB_SEARCH_PREFIX)
        .map(|s| s.to_string())
        .unwrap_or(text);

    let query = query.split_whitespace().collect::<Vec<_>>().join(" ");
    if query.is_empty() { None } else { Some(query) }
}

/// 生成22位大小写字母和数字的随机字符串
fn generate_random_id_22() -> String {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    (0..22)
        .map(|_| {
            let idx = fastrand::usize(..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

/// 生成8位小写字母和数字的随机字符串
fn generate_random_id_8() -> String {
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    (0..8)
        .map(|_| {
            let idx = fastrand::usize(..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

/// 创建 MCP 请求
///
/// ID 格式: web_search_tooluse_{22位随机}_{毫秒时间戳}_{8位随机}
pub fn create_mcp_request(query: &str) -> (String, McpRequest) {
    let random_22 = generate_random_id_22();
    let timestamp = chrono::Utc::now().timestamp_millis();
    let random_8 = generate_random_id_8();

    let request_id = format!(
        "web_search_tooluse_{}_{}_{}",
        random_22, timestamp, random_8
    );

    // tool_use_id 使用相同格式
    let tool_use_id = format!(
        "srvtoolu_{}",
        &Uuid::new_v4().to_string().replace('-', "")[..32]
    );

    let request = McpRequest {
        id: request_id,
        jsonrpc: "2.0".to_string(),
        method: "tools/call".to_string(),
        params: McpParams {
            name: "web_search".to_string(),
            arguments: McpArguments {
                query: query.to_string(),
            },
        },
    };

    (tool_use_id, request)
}

/// 解析 MCP 响应中的搜索结果
pub fn parse_search_results(mcp_response: &McpResponse) -> Option<WebSearchResults> {
    let result = mcp_response.result.as_ref()?;
    let content = result.content.first()?;

    if content.content_type != "text" {
        return None;
    }

    serde_json::from_str(&content.text).ok()
}

/// 生成 WebSearch SSE 响应流
pub fn create_websearch_sse_stream(
    model: String,
    query: String,
    tool_use_id: String,
    search_results: Option<WebSearchResults>,
    input_tokens: i32,
    cache_context: Option<WebSearchCacheContext>,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    let events = generate_websearch_events(
        &model,
        &query,
        &tool_use_id,
        search_results,
        input_tokens,
        cache_context,
    );

    stream::iter(
        events
            .into_iter()
            .map(|e| Ok(Bytes::from(e.to_sse_string()))),
    )
}

/// 生成 WebSearch SSE 事件序列
fn generate_websearch_events(
    model: &str,
    query: &str,
    tool_use_id: &str,
    search_results: Option<WebSearchResults>,
    input_tokens: i32,
    cache_context: Option<WebSearchCacheContext>,
) -> Vec<SseEvent> {
    let mut events = Vec::new();
    let message_id = format!("msg_{}", &Uuid::new_v4().to_string().replace('-', "")[..24]);
    let billed_input_tokens = cache_context
        .map(|ctx| {
            billed_input_tokens(
                input_tokens,
                ctx.cache_creation_input_tokens,
                ctx.cache_read_input_tokens,
            )
        })
        .unwrap_or(input_tokens);

    // 1. message_start
    // 本地 WebSearch 的 usage 需要沿用外层预先计算的统计口径。
    let mut message_start_usage = json!({
        "input_tokens": billed_input_tokens,
        "output_tokens": 0,
    });
    if let Some(cache_context) = cache_context {
        message_start_usage["cache_creation_input_tokens"] =
            json!(cache_context.cache_creation_input_tokens);
        message_start_usage["cache_read_input_tokens"] =
            json!(cache_context.cache_read_input_tokens);
    }
    events.push(SseEvent::new(
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "usage": message_start_usage
            }
        }),
    ));

    // 2. content_block_start (text - 搜索决策说明, index 0)
    let decision_text = format!("I'll search for \"{}\".", query);
    events.push(SseEvent::new(
        "content_block_start",
        json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": ""
            }
        }),
    ));

    events.push(SseEvent::new(
        "content_block_delta",
        json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": decision_text
            }
        }),
    ));

    events.push(SseEvent::new(
        "content_block_stop",
        json!({
            "type": "content_block_stop",
            "index": 0
        }),
    ));

    // 3. content_block_start (server_tool_use, index 1)
    // server_tool_use 是服务端工具，input 在 content_block_start 中一次性完整发送，
    // 不像客户端 tool_use 需要通过 input_json_delta 增量传输。
    events.push(SseEvent::new(
        "content_block_start",
        json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "id": tool_use_id,
                "type": "server_tool_use",
                "name": "web_search",
                "input": {"query": query}
            }
        }),
    ));

    // 4. content_block_stop (server_tool_use)
    events.push(SseEvent::new(
        "content_block_stop",
        json!({
            "type": "content_block_stop",
            "index": 1
        }),
    ));

    // 5. content_block_start (web_search_tool_result, index 2)
    // 官方 API 的 web_search_tool_result 没有 tool_use_id 字段
    let search_content = if let Some(ref results) = search_results {
        results
            .results
            .iter()
            .map(|r| {
                let page_age = r.published_date.and_then(|ms| {
                    chrono::DateTime::from_timestamp_millis(ms)
                        .map(|dt| dt.format("%B %-d, %Y").to_string())
                });
                json!({
                    "type": "web_search_result",
                    "title": r.title,
                    "url": r.url,
                    "encrypted_content": r.snippet.clone().unwrap_or_default(),
                    "page_age": page_age
                })
            })
            .collect::<Vec<_>>()
    } else {
        vec![]
    };

    events.push(SseEvent::new(
        "content_block_start",
        json!({
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "web_search_tool_result",
                "content": search_content
            }
        }),
    ));

    // 6. content_block_stop (web_search_tool_result)
    events.push(SseEvent::new(
        "content_block_stop",
        json!({
            "type": "content_block_stop",
            "index": 2
        }),
    ));

    // 7. content_block_start (text, index 3)
    events.push(SseEvent::new(
        "content_block_start",
        json!({
            "type": "content_block_start",
            "index": 3,
            "content_block": {
                "type": "text",
                "text": ""
            }
        }),
    ));

    // 8. content_block_delta (text_delta) - 生成搜索结果摘要
    let summary = generate_search_summary(query, &search_results);

    // 分块发送文本
    let chunk_size = 100;
    for chunk in summary.chars().collect::<Vec<_>>().chunks(chunk_size) {
        let text: String = chunk.iter().collect();
        events.push(SseEvent::new(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 3,
                "delta": {
                    "type": "text_delta",
                    "text": text
                }
            }),
        ));
    }

    // 9. content_block_stop (text)
    events.push(SseEvent::new(
        "content_block_stop",
        json!({
            "type": "content_block_stop",
            "index": 3
        }),
    ));

    // 10. message_delta
    // 官方 API 的 message_delta.delta 中没有 stop_sequence 字段
    let output_tokens = (summary.len() as i32 + 3) / 4; // 简单估算
    let mut message_delta_usage = json!({
        "input_tokens": billed_input_tokens,
        "output_tokens": output_tokens,
        "server_tool_use": {
            "web_search_requests": 1
        }
    });
    if let Some(cache_context) = cache_context {
        message_delta_usage["cache_creation_input_tokens"] =
            json!(cache_context.cache_creation_input_tokens);
        message_delta_usage["cache_read_input_tokens"] =
            json!(cache_context.cache_read_input_tokens);
    }
    events.push(SseEvent::new(
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn"
            },
            "usage": message_delta_usage
        }),
    ));

    // 11. message_stop
    events.push(SseEvent::new(
        "message_stop",
        json!({
            "type": "message_stop"
        }),
    ));

    events
}

/// 生成搜索结果摘要
fn generate_search_summary(query: &str, results: &Option<WebSearchResults>) -> String {
    let mut summary = format!("Here are the search results for \"{}\":\n\n", query);

    if let Some(results) = results {
        for (i, result) in results.results.iter().enumerate() {
            summary.push_str(&format!("{}. **{}**\n", i + 1, result.title));
            if let Some(ref snippet) = result.snippet {
                // 截断过长的摘要（安全处理 UTF-8 多字节字符）
                let truncated = match snippet.char_indices().nth(200) {
                    Some((idx, _)) => format!("{}...", &snippet[..idx]),
                    None => snippet.clone(),
                };
                summary.push_str(&format!("   {}\n", truncated));
            }
            summary.push_str(&format!("   Source: {}\n\n", result.url));
        }
    } else {
        summary.push_str("No results found.\n");
    }

    summary.push_str("\nPlease note that these are web search results and may not be fully accurate or up-to-date.");

    summary
}

/// 处理 WebSearch 请求
pub async fn handle_websearch_request(
    provider: std::sync::Arc<crate::kiro::provider::KiroProvider>,
    payload: &MessagesRequest,
    cache_tracker: Option<&std::sync::Arc<crate::anthropic::cache_tracker::CacheTracker>>,
    cache_profile: Option<&crate::anthropic::cache_tracker::CacheProfile>,
    input_tokens: i32,
) -> Response {
    // 1. 提取搜索查询
    let query = match extract_search_query(payload) {
        Some(q) => q,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(
                    "invalid_request_error",
                    "无法从消息中提取搜索查询",
                )),
            )
                .into_response();
        }
    };

    tracing::info!(query = %query, "处理 WebSearch 请求");

    // 2. 创建 MCP 请求
    let (tool_use_id, mcp_request) = create_mcp_request(&query);

    // 3. 调用 Kiro MCP API
    let (search_results, final_cache_context) = match call_mcp_api(&provider, &mcp_request).await {
        Ok(api_result) => {
            let resolved_cache_context = match (cache_tracker, cache_profile) {
                (Some(cache_tracker), Some(cache_profile)) => {
                    let resolved_cache_context =
                        resolve_cache_usage(cache_tracker, api_result.credential_id, cache_profile);
                    tracing::info!(
                        credential_id = api_result.credential_id,
                        final_cache_creation_input_tokens =
                            resolved_cache_context.cache_creation_input_tokens,
                        final_cache_read_input_tokens =
                            resolved_cache_context.cache_read_input_tokens,
                        "Resolved cache usage for websearch request"
                    );
                    cache_tracker.update(api_result.credential_id, cache_profile);
                    Some(resolved_cache_context)
                }
                _ => None,
            };
            (
                parse_search_results(&api_result.response),
                resolved_cache_context,
            )
        }
        Err(e) => {
            tracing::warn!("MCP API 调用失败: {}", e);
            (None, None)
        }
    };

    // 4. 生成 SSE 响应
    let model = payload.model.clone();
    if payload.stream {
        let stream = create_websearch_sse_stream(
            model,
            query,
            tool_use_id,
            search_results,
            input_tokens,
            final_cache_context,
        );

        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .body(Body::from_stream(stream))
            .unwrap();
    }

    let summary = generate_search_summary(&query, &search_results);
    let search_content = if let Some(ref results) = search_results {
        results
            .results
            .iter()
            .map(|r| {
                json!({
                    "type": "web_search_result",
                    "title": r.title,
                    "url": r.url,
                    "encrypted_content": r.snippet.clone().unwrap_or_default(),
                    "page_age": null
                })
            })
            .collect::<Vec<_>>()
    } else {
        vec![]
    };

    let output_tokens = (summary.len() as i32 + 3) / 4; // 简单估算
    let billed_input_tokens = final_cache_context
        .map(|ctx| {
            billed_input_tokens(
                input_tokens,
                ctx.cache_creation_input_tokens,
                ctx.cache_read_input_tokens,
            )
        })
        .unwrap_or(input_tokens);

    let mut usage = json!({
        "input_tokens": billed_input_tokens,
        "output_tokens": output_tokens,
    });
    if let Some(final_cache_context) = final_cache_context {
        usage["cache_creation_input_tokens"] =
            json!(final_cache_context.cache_creation_input_tokens);
        usage["cache_read_input_tokens"] = json!(final_cache_context.cache_read_input_tokens);
    }

    let response_body = json!({
        "id": format!("msg_{}", &Uuid::new_v4().to_string().replace('-', "")[..24]),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "server_tool_use",
                "id": tool_use_id,
                "name": "web_search",
                "input": { "query": query }
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": tool_use_id,
                "content": search_content
            },
            {
                "type": "text",
                "text": summary
            }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": usage
    });

    (StatusCode::OK, Json(response_body)).into_response()
}

struct ParsedMcpCallResult {
    response: McpResponse,
    credential_id: u64,
}

/// 调用 Kiro MCP API
async fn call_mcp_api(
    provider: &crate::kiro::provider::KiroProvider,
    request: &McpRequest,
) -> anyhow::Result<ParsedMcpCallResult> {
    let request_body = serde_json::to_string(request)?;

    tracing::debug!("MCP request: {}", request_body);

    let api_result = provider.call_mcp(&request_body).await?;

    let body = api_result.response.text().await?;
    tracing::debug!("MCP response: {}", body);

    let mcp_response: McpResponse = serde_json::from_str(&body)?;

    if let Some(ref error) = mcp_response.error {
        anyhow::bail!(
            "MCP error: {} - {}",
            error.code.unwrap_or(-1),
            error.message.as_deref().unwrap_or("Unknown error")
        );
    }

    Ok(ParsedMcpCallResult {
        response: mcp_response,
        credential_id: api_result.credential_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_cache_usage_uses_real_credential_id() {
        use crate::anthropic::cache_tracker::CacheTracker;
        use crate::anthropic::types::{CacheControl, Message, SystemMessage};

        let long_text = "This is a cached websearch system block. ".repeat(100);
        let payload = MessagesRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!("Perform a web search for the query: rust cache"),
            }],
            stream: false,
            system: Some(vec![SystemMessage {
                text: format!("{}{}", long_text, long_text),
                block_type: Some("text".to_string()),
                cache_control: Some(CacheControl {
                    cache_type: "ephemeral".to_string(),
                    ttl: None,
                }),
            }]),
            tools: None,
            tool_choice: None,
            thinking: None,
            output_config: None,
            metadata: None,
        };

        let tracker = CacheTracker::new(std::time::Duration::from_secs(300));
        let total = crate::token::count_all_tokens(
            payload.model.clone(),
            payload.system.clone(),
            payload.messages.clone(),
            payload.tools.clone(),
        ) as i32;
        let profile = tracker.build_profile(&payload, total);

        let initial = resolve_cache_usage(&tracker, 7, &profile);
        assert_eq!(initial.cache_read_input_tokens, 0);

        tracker.update(7, &profile);
        let resolved = resolve_cache_usage(&tracker, 7, &profile);
        assert!(resolved.cache_read_input_tokens > 0);
    }

    #[test]
    fn test_generate_websearch_events_uses_raw_usage_and_cache_context() {
        let events = generate_websearch_events(
            "claude-sonnet-4",
            "rust",
            "srvtoolu_test",
            None,
            123,
            Some(WebSearchCacheContext {
                cache_creation_input_tokens: 7,
                cache_read_input_tokens: 9,
                cache_creation_5m_input_tokens: 3,
                cache_creation_1h_input_tokens: 4,
            }),
        );

        let message_start = events
            .iter()
            .find(|e| e.event == "message_start")
            .expect("should have message_start");
        let message_delta = events
            .iter()
            .find(|e| e.event == "message_delta")
            .expect("should have message_delta");

        assert_eq!(message_start.data["message"]["usage"]["input_tokens"], 107);
        assert_eq!(
            message_start.data["message"]["usage"]["cache_creation_input_tokens"],
            7
        );
        assert_eq!(
            message_start.data["message"]["usage"]["cache_read_input_tokens"],
            9
        );

        assert_eq!(message_delta.data["usage"]["input_tokens"], 107);
        assert_eq!(
            message_delta.data["usage"]["cache_creation_input_tokens"],
            7
        );
        assert_eq!(message_delta.data["usage"]["cache_read_input_tokens"], 9);
    }

    #[test]
    fn test_generate_websearch_events_omits_cache_usage_when_disabled() {
        let events =
            generate_websearch_events("claude-sonnet-4", "rust", "srvtoolu_test", None, 123, None);

        let message_start = events
            .iter()
            .find(|e| e.event == "message_start")
            .expect("should have message_start");
        let message_delta = events
            .iter()
            .find(|e| e.event == "message_delta")
            .expect("should have message_delta");

        assert_eq!(message_start.data["message"]["usage"]["input_tokens"], 123);
        assert!(
            message_start.data["message"]["usage"]
                .get("cache_creation_input_tokens")
                .is_none()
        );
        assert!(
            message_delta.data["usage"]
                .get("cache_read_input_tokens")
                .is_none()
        );
        assert!(message_delta.data["usage"].get("cache_creation").is_none());
    }

    #[test]
    fn test_has_web_search_tool_only_one() {
        use crate::anthropic::types::{Message, Tool};

        let req = MessagesRequest {
            model: "claude-sonnet-4".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!("test"),
            }],
            stream: true,
            system: None,
            tools: Some(vec![Tool {
                tool_type: Some("web_search_20250305".to_string()),
                name: "web_search".to_string(),
                description: String::new(),
                input_schema: Default::default(),
                max_uses: Some(8),
                cache_control: None,
            }]),
            tool_choice: None,
            thinking: None,
            output_config: None,
            metadata: None,
        };

        assert!(has_web_search_tool(&req));
    }

    #[test]
    fn test_has_web_search_tool_when_multiple_tools() {
        use crate::anthropic::types::{Message, Tool};

        let req = MessagesRequest {
            model: "claude-sonnet-4".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!("test"),
            }],
            stream: true,
            system: None,
            tools: Some(vec![
                Tool {
                    tool_type: Some("web_search_20250305".to_string()),
                    name: "web_search".to_string(),
                    description: String::new(),
                    input_schema: Default::default(),
                    max_uses: Some(8),
                    cache_control: None,
                },
                Tool {
                    tool_type: None,
                    name: "other_tool".to_string(),
                    description: "Other tool".to_string(),
                    input_schema: Default::default(),
                    max_uses: None,
                    cache_control: None,
                },
            ]),
            tool_choice: None,
            thinking: None,
            output_config: None,
            metadata: None,
        };

        assert!(has_web_search_tool(&req));
    }

    #[test]
    fn test_has_web_search_tool_when_name_missing_but_type_matches() {
        use crate::anthropic::types::{Message, Tool};

        let req = MessagesRequest {
            model: "claude-sonnet-4".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!("test"),
            }],
            stream: true,
            system: None,
            tools: Some(vec![Tool {
                tool_type: Some("web_search_20250305".to_string()),
                name: "".to_string(), // 模拟客户端只传 type 的情况
                description: String::new(),
                input_schema: Default::default(),
                max_uses: Some(8),
                cache_control: None,
            }]),
            tool_choice: None,
            thinking: None,
            output_config: None,
            metadata: None,
        };

        assert!(has_web_search_tool(&req));
    }

    #[test]
    fn test_extract_search_query_with_prefix() {
        use crate::anthropic::types::Message;

        let req = MessagesRequest {
            model: "claude-sonnet-4".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!([{
                    "type": "text",
                    "text": "Perform a web search for the query: rust latest version 2026"
                }]),
            }],
            stream: true,
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            output_config: None,
            metadata: None,
        };

        let query = extract_search_query(&req);
        // 前缀应该被去除
        assert_eq!(query, Some("rust latest version 2026".to_string()));
    }

    #[test]
    fn test_extract_search_query_plain_text() {
        use crate::anthropic::types::Message;

        let req = MessagesRequest {
            model: "claude-sonnet-4".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::json!("What is the weather today?"),
            }],
            stream: true,
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            output_config: None,
            metadata: None,
        };

        let query = extract_search_query(&req);
        assert_eq!(query, Some("What is the weather today?".to_string()));
    }

    #[test]
    fn test_extract_search_query_uses_last_user_message() {
        use crate::anthropic::types::Message;

        let req = MessagesRequest {
            model: "claude-sonnet-4".to_string(),
            max_tokens: 1024,
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: serde_json::json!("not a query"),
                },
                Message {
                    role: "assistant".to_string(),
                    content: serde_json::json!("ok"),
                },
                Message {
                    role: "user".to_string(),
                    content: serde_json::json!([{
                        "type": "text",
                        "text": "Perform a web search for the query: rust"
                    }]),
                },
            ],
            stream: true,
            system: None,
            tools: None,
            tool_choice: None,
            thinking: None,
            output_config: None,
            metadata: None,
        };

        let query = extract_search_query(&req);
        assert_eq!(query, Some("rust".to_string()));
    }

    #[test]
    fn test_create_mcp_request() {
        let (tool_use_id, request) = create_mcp_request("test query");

        assert!(tool_use_id.starts_with("srvtoolu_"));
        assert_eq!(request.jsonrpc, "2.0");
        assert_eq!(request.method, "tools/call");
        assert_eq!(request.params.name, "web_search");
        assert_eq!(request.params.arguments.query, "test query");

        // 验证 ID 格式: web_search_tooluse_{22位}_{时间戳}_{8位}
        assert!(request.id.starts_with("web_search_tooluse_"));
    }

    #[test]
    fn test_mcp_request_id_format() {
        let (_, request) = create_mcp_request("test");

        // 格式: web_search_tooluse_{22位}_{毫秒时间戳}_{8位}
        let id = &request.id;
        assert!(id.starts_with("web_search_tooluse_"));

        let suffix = &id["web_search_tooluse_".len()..];
        let parts: Vec<&str> = suffix.split('_').collect();
        assert_eq!(parts.len(), 3, "应该有3个部分: 22位随机_时间戳_8位随机");

        // 第一部分: 22位大小写字母和数字
        assert_eq!(parts[0].len(), 22);
        assert!(parts[0].chars().all(|c| c.is_ascii_alphanumeric()));

        // 第二部分: 毫秒时间戳
        assert!(parts[1].parse::<i64>().is_ok());

        // 第三部分: 8位小写字母和数字
        assert_eq!(parts[2].len(), 8);
        assert!(
            parts[2]
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        );
    }

    #[test]
    fn test_parse_search_results() {
        let response = McpResponse {
            error: None,
            id: "test_id".to_string(),
            jsonrpc: "2.0".to_string(),
            result: Some(McpResult {
                content: vec![McpContent {
                    content_type: "text".to_string(),
                    text: r#"{"results":[{"title":"Test","url":"https://example.com","snippet":"Test snippet"}],"totalResults":1}"#.to_string(),
                }],
                is_error: false,
            }),
        };

        let results = parse_search_results(&response);
        assert!(results.is_some());
        let results = results.unwrap();
        assert_eq!(results.results.len(), 1);
        assert_eq!(results.results[0].title, "Test");
    }

    #[test]
    fn test_generate_search_summary() {
        let results = WebSearchResults {
            results: vec![WebSearchResult {
                title: "Test Result".to_string(),
                url: "https://example.com".to_string(),
                snippet: Some("This is a test snippet".to_string()),
                published_date: None,
                id: None,
                domain: None,
                max_verbatim_word_limit: None,
                public_domain: None,
            }],
            total_results: Some(1),
            query: Some("test".to_string()),
            error: None,
        };

        let summary = generate_search_summary("test", &Some(results));

        assert!(summary.contains("Test Result"));
        assert!(summary.contains("https://example.com"));
        assert!(summary.contains("This is a test snippet"));
    }
}
