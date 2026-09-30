//! Server-side `web_search` for the Anthropic `/v1/messages` surface.
//!
//! Anthropic's `web_search_*` tool is a SERVER tool: the API runs the searches
//! itself and answers with `server_tool_use` + `web_search_tool_result` blocks,
//! so a client (Claude Code's WebSearch tool, an SDK agent) never sees a tool
//! call to execute. mlx-serve renders the tool to the model as an ordinary
//! `web_search(query)` function (`TOOL_DESCRIPTION` / `TOOL_PARAMETERS`, used by
//! server.zig's `buildOpenAIToolsJson`), runs the request through the normal
//! /v1/messages handler with its output captured (`Capture`, fed from the
//! `Conn.capture` hook), answers each call from a SearXNG instance
//! (`MLX_SERVE_WEB_SEARCH_URL`, its JSON API) and loops until the model answers
//! without searching. A streaming client gets the model's blocks live through
//! `Relay`, renumbered across the inner rounds; the search blocks are emitted
//! as each round completes.
//!
//! This file holds the self-contained parts (config, SearXNG client + parsing,
//! SSE capture, output rendering); the loop is `handleAnthropicWebSearch` in
//! server.zig.

const std = @import("std");
const lan = @import("lan.zig");
const log = @import("log.zig");
const writeJsonString = @import("ollama.zig").writeJsonString;

const Allocator = std.mem.Allocator;

/// Searches per request when the tool does not set `max_uses`.
pub const DEFAULT_MAX_USES: u32 = 5;
/// Results handed to the model (and returned to the client) per search.
pub const MAX_RESULTS: usize = 8;
/// Anthropic rejects longer queries with `query_too_long`; so do we.
pub const MAX_QUERY_LEN: usize = 500;
/// Per-result snippet cap in the model's tool result, in bytes.
pub const SNIPPET_MAX: usize = 700;
/// Thinking-block signature, the same one the /v1/messages handlers emit.
pub const SIGNATURE = "mlx-serve-local";
/// `encrypted_content` is opaque to clients; ours is the snippet, base64'd behind this tag.
pub const ENCRYPTED_PREFIX = "mlxs1:";
/// Env var naming the SearXNG base URL, e.g. `http://127.0.0.1:8888`.
pub const URL_ENV = "MLX_SERVE_WEB_SEARCH_URL";

pub const TOOL_DESCRIPTION =
    "Search the web for current information. Returns the top results with title, URL and a text snippet. " ++
    "Use it when the answer depends on recent events or on facts you are not certain of; " ++
    "refine the query and search again if the first results are not enough.";
pub const TOOL_PARAMETERS =
    \\{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]}
;

const PING = "{\"type\":\"ping\"}";

// ── Tool config ─────────────────────────────────────────────────────────────

/// True for an Anthropic `web_search_*` server-tool definition.
pub fn isServerTool(tool: std.json.ObjectMap) bool {
    const t = tool.get("type") orelse return false;
    return t == .string and std.mem.startsWith(u8, t.string, "web_search_");
}

pub const ToolConfig = struct {
    /// The function name the model calls (the tool's `name`, normally "web_search").
    name: []const u8,
    max_uses: u32 = DEFAULT_MAX_USES,
    allowed_domains: []const []const u8 = &.{},
    blocked_domains: []const []const u8 = &.{},
};

/// The first `web_search_*` server tool in `tools`, or null. Strings borrow
/// from the request JSON; the domain lists are allocated with `arena`.
pub fn findServerTool(arena: Allocator, tools: []const std.json.Value) !?ToolConfig {
    for (tools) |tv| {
        if (tv != .object or !isServerTool(tv.object)) continue;
        const o = tv.object;
        var cfg = ToolConfig{ .name = "web_search" };
        if (o.get("name")) |n| if (n == .string and n.string.len > 0) {
            cfg.name = n.string;
        };
        if (o.get("max_uses")) |m| if (m == .integer and m.integer > 0) {
            cfg.max_uses = @intCast(@min(m.integer, 50));
        };
        cfg.allowed_domains = try stringList(arena, o.get("allowed_domains"));
        cfg.blocked_domains = try stringList(arena, o.get("blocked_domains"));
        return cfg;
    }
    return null;
}

fn stringList(arena: Allocator, v: ?std.json.Value) ![]const []const u8 {
    const val = v orelse return &.{};
    if (val != .array) return &.{};
    var out = std.ArrayList([]const u8).empty;
    for (val.array.items) |item| {
        if (item == .string and item.string.len > 0) try out.append(arena, item.string);
    }
    return out.items;
}

// ── Domain filters ──────────────────────────────────────────────────────────

/// Host part of a URL, lowercase-insensitive comparisons are the caller's job.
fn hostOf(url: []const u8) []const u8 {
    var rest = url;
    if (std.mem.indexOf(u8, rest, "://")) |p| rest = rest[p + 3 ..];
    const end = std.mem.indexOfAny(u8, rest, "/?#") orelse rest.len;
    rest = rest[0..end];
    if (std.mem.lastIndexOfScalar(u8, rest, '@')) |at| rest = rest[at + 1 ..];
    if (std.mem.lastIndexOfScalar(u8, rest, ':')) |c| rest = rest[0..c];
    return rest;
}

/// `host` is `domain` or a subdomain of it. A configured domain may carry a
/// scheme, a path or a leading "www.", which are ignored.
fn domainMatches(host: []const u8, domain_raw: []const u8) bool {
    var d = hostOf(domain_raw);
    if (std.ascii.startsWithIgnoreCase(d, "www.")) d = d[4..];
    if (d.len == 0) return false;
    if (std.ascii.eqlIgnoreCase(host, d)) return true;
    return host.len > d.len and host[host.len - d.len - 1] == '.' and
        std.ascii.eqlIgnoreCase(host[host.len - d.len ..], d);
}

pub fn domainAllowed(url: []const u8, cfg: ToolConfig) bool {
    const host = hostOf(url);
    for (cfg.blocked_domains) |d| if (domainMatches(host, d)) return false;
    if (cfg.allowed_domains.len == 0) return true;
    for (cfg.allowed_domains) |d| if (domainMatches(host, d)) return true;
    return false;
}

// ── SearXNG client ──────────────────────────────────────────────────────────

pub const Endpoint = struct {
    ip4: [4]u8,
    port: u16,
    /// Path prefix without a trailing slash ("" for a root install).
    base_path: []const u8,
};

/// `http://<ipv4|localhost>[:port][/prefix]`. HTTPS and DNS names are not
/// supported: the instance is meant to run next to the server.
pub fn parseEndpoint(url: []const u8) ?Endpoint {
    if (!std.mem.startsWith(u8, url, "http://")) return null;
    const rest = url["http://".len..];
    const slash = std.mem.indexOfScalar(u8, rest, '/') orelse rest.len;
    const hostport = rest[0..slash];
    const base_path = std.mem.trimEnd(u8, rest[slash..], "/");
    var host = hostport;
    var port: u16 = 80;
    if (std.mem.lastIndexOfScalar(u8, hostport, ':')) |c| {
        host = hostport[0..c];
        port = std.fmt.parseInt(u16, hostport[c + 1 ..], 10) catch return null;
    }
    var ip4: [4]u8 = undefined;
    if (std.ascii.eqlIgnoreCase(host, "localhost")) {
        ip4 = .{ 127, 0, 0, 1 };
    } else {
        var it = std.mem.splitScalar(u8, host, '.');
        var i: usize = 0;
        while (it.next()) |part| : (i += 1) {
            if (i >= 4) return null;
            ip4[i] = std.fmt.parseInt(u8, part, 10) catch return null;
        }
        if (i != 4) return null;
    }
    return .{ .ip4 = ip4, .port = port, .base_path = base_path };
}

/// The configured endpoint, or null when `MLX_SERVE_WEB_SEARCH_URL` is unset
/// or unparseable (then every search answers `unavailable`).
pub fn endpointFromEnv() ?Endpoint {
    const z = std.c.getenv(URL_ENV) orelse return null;
    const url = std.mem.span(z);
    return parseEndpoint(url) orelse {
        log.warn("[web_search] {s}={s} is not http://<ipv4|localhost>:<port>[/path]; searches will fail\n", .{ URL_ENV, url });
        return null;
    };
}

/// application/x-www-form-urlencoded query value.
fn appendQueryEscaped(allocator: Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    const hex = "0123456789ABCDEF";
    for (s) |c| {
        if (std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.' or c == '~') {
            try out.append(allocator, c);
        } else if (c == ' ') {
            try out.append(allocator, '+');
        } else {
            try out.appendSlice(allocator, &.{ '%', hex[c >> 4], hex[c & 15] });
        }
    }
}

pub fn searchPath(allocator: Allocator, ep: Endpoint, query: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, ep.base_path);
    try out.appendSlice(allocator, "/search?q=");
    try appendQueryEscaped(allocator, &out, query);
    try out.appendSlice(allocator, "&format=json");
    return out.toOwnedSlice(allocator);
}

pub const Result = struct {
    title: []const u8,
    url: []const u8,
    snippet: []const u8,
    page_age: ?[]const u8 = null,
};

/// Anthropic's `web_search_tool_result_error` codes.
pub const ErrorCode = enum { unavailable, invalid_input, max_uses_exceeded, query_too_long, too_many_requests };

pub const Outcome = union(enum) {
    results: []const Result,
    failed: ErrorCode,
};

/// Run one search. Never fails: transport and parse problems become `.unavailable`.
pub fn search(arena: Allocator, ep: Endpoint, query: []const u8, cfg: ToolConfig) Outcome {
    if (query.len == 0) return .{ .failed = .invalid_input };
    if (query.len > MAX_QUERY_LEN) return .{ .failed = .query_too_long };
    const raw = fetch(arena, ep, query) catch |err| {
        log.warn("[web_search] SearXNG request failed: {s}\n", .{@errorName(err)});
        return .{ .failed = .unavailable };
    };
    const resp = parseHttpResponse(arena, raw) catch |err| {
        log.warn("[web_search] bad SearXNG response: {s}\n", .{@errorName(err)});
        return .{ .failed = .unavailable };
    };
    if (resp.status == 429) return .{ .failed = .too_many_requests };
    if (resp.status != 200) {
        log.warn("[web_search] SearXNG answered {d} (is `json` in search.formats?)\n", .{resp.status});
        return .{ .failed = .unavailable };
    }
    const results = parseSearxResults(arena, resp.body, cfg, MAX_RESULTS) catch |err| {
        log.warn("[web_search] unparseable SearXNG JSON: {s}\n", .{@errorName(err)});
        return .{ .failed = .unavailable };
    };
    return .{ .results = results };
}

fn fetch(arena: Allocator, ep: Endpoint, query: []const u8) ![]u8 {
    const path = try searchPath(arena, ep, query);
    const fd = try lan.connectTimeout(ep.ip4, ep.port, 3000);
    defer _ = std.c.close(fd);
    // SearXNG's own engine timeouts are a few seconds; this only bounds a wedged instance.
    const tv = std.c.timeval{ .sec = 20, .usec = 0 };
    std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&tv)) catch {};
    const head = try std.fmt.allocPrint(arena, "GET {s} HTTP/1.1\r\nHost: {d}.{d}.{d}.{d}:{d}\r\nAccept: application/json\r\nUser-Agent: mlx-serve\r\nConnection: close\r\n\r\n", .{
        path, ep.ip4[0], ep.ip4[1], ep.ip4[2], ep.ip4[3], ep.port,
    });
    try lan.writeAllFd(fd, head);
    var resp = std.ArrayList(u8).empty;
    var chunk: [16 * 1024]u8 = undefined;
    while (resp.items.len < 8 * 1024 * 1024) {
        const n = try lan.readFd(fd, &chunk);
        if (n == 0) break;
        try resp.appendSlice(arena, chunk[0..n]);
    }
    return resp.items;
}

const HttpResponse = struct { status: u16, body: []const u8 };

fn parseHttpResponse(arena: Allocator, raw: []const u8) !HttpResponse {
    const he = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return error.NoHead;
    const head = raw[0..he];
    if (!std.mem.startsWith(u8, head, "HTTP/1.") or head.len < 12) return error.NoStatus;
    const status = std.fmt.parseInt(u16, head[9..12], 10) catch return error.NoStatus;
    var body = raw[he + 4 ..];
    if (lan.headerValueCI(head, "transfer-encoding")) |te| {
        if (std.ascii.findIgnoreCase(te, "chunked") != null) body = try dechunk(arena, body);
    } else if (lan.headerValueCI(head, "content-length")) |cl| {
        const n = std.fmt.parseInt(usize, cl, 10) catch body.len;
        if (n < body.len) body = body[0..n];
    }
    return .{ .status = status, .body = body };
}

fn dechunk(arena: Allocator, body: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    var rest = body;
    while (true) {
        const le = std.mem.indexOf(u8, rest, "\r\n") orelse return error.BadChunk;
        const line = rest[0..le];
        const size_str = std.mem.trim(u8, line[0 .. std.mem.indexOfScalar(u8, line, ';') orelse line.len], " ");
        const size = std.fmt.parseInt(usize, size_str, 16) catch return error.BadChunk;
        rest = rest[le + 2 ..];
        if (size == 0) break;
        if (rest.len < size) return error.BadChunk;
        try out.appendSlice(arena, rest[0..size]);
        rest = rest[size..];
        if (std.mem.startsWith(u8, rest, "\r\n")) rest = rest[2..];
    }
    return out.items;
}

/// Byte length of the longest prefix of `s` that is at most `max` bytes and ends on a UTF-8 boundary.
fn utf8Prefix(s: []const u8, max: usize) usize {
    if (s.len <= max) return s.len;
    var n = max;
    while (n > 0 and (s[n] & 0xC0) == 0x80) n -= 1;
    return n;
}

fn collapseSpace(arena: Allocator, s: []const u8) ![]const u8 {
    var out = std.ArrayList(u8).empty;
    var space = false;
    for (std.mem.trim(u8, s, " \t\r\n")) |c| {
        if (c == ' ' or c == '\t' or c == '\r' or c == '\n') {
            space = true;
            continue;
        }
        if (space) try out.append(arena, ' ');
        space = false;
        try out.append(arena, c);
    }
    return out.items;
}

fn strField(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

/// SearXNG `format=json` → at most `max` results that pass the domain filters.
pub fn parseSearxResults(arena: Allocator, body: []const u8, cfg: ToolConfig, max: usize) ![]const Result {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{});
    if (parsed != .object) return error.NotAnObject;
    const arr = parsed.object.get("results") orelse return error.NoResults;
    if (arr != .array) return error.NoResults;
    var out = std.ArrayList(Result).empty;
    for (arr.array.items) |item| {
        if (out.items.len >= max) break;
        if (item != .object) continue;
        const url = strField(item.object, "url") orelse continue;
        if (!std.mem.startsWith(u8, url, "http://") and !std.mem.startsWith(u8, url, "https://")) continue;
        if (!domainAllowed(url, cfg)) continue;
        var dup = false;
        for (out.items) |r| if (std.mem.eql(u8, r.url, url)) {
            dup = true;
        };
        if (dup) continue;
        const title = try collapseSpace(arena, strField(item.object, "title") orelse url);
        const content = try collapseSpace(arena, strField(item.object, "content") orelse "");
        var page_age: ?[]const u8 = null;
        if (strField(item.object, "publishedDate")) |d| {
            if (d.len > 0) page_age = if (d.len >= 10) d[0..10] else d;
        }
        try out.append(arena, .{
            .title = if (title.len > 0) title else url,
            .url = url,
            .snippet = content[0..utf8Prefix(content, SNIPPET_MAX)],
            .page_age = page_age,
        });
    }
    return out.items;
}

// ── Rendering ───────────────────────────────────────────────────────────────

/// The tool-result text the model reads for one search.
pub fn modelText(arena: Allocator, query: []const u8, outcome: Outcome) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(arena);
    switch (outcome) {
        .results => |rs| {
            if (rs.len == 0) {
                try w.writer.print("No web results found for \"{s}\". Try a different query, or answer from what you know and say that the search found nothing.", .{query});
            } else {
                try w.writer.print("Web search results for \"{s}\":\n", .{query});
                for (rs, 1..) |r, i| {
                    try w.writer.print("\n[{d}] {s}\nURL: {s}\n", .{ i, r.title, r.url });
                    if (r.page_age) |age| try w.writer.print("Published: {s}\n", .{age});
                    if (r.snippet.len > 0) try w.writer.print("{s}\n", .{r.snippet});
                }
            }
        },
        .failed => |code| switch (code) {
            .max_uses_exceeded => try w.writer.writeAll("Search limit reached for this request. Do not search again; answer with the results already gathered."),
            .query_too_long => try w.writer.writeAll("The query is too long. Search again with a shorter query."),
            .invalid_input => try w.writer.writeAll("The search call had no usable \"query\" string. Call web_search with {\"query\": \"...\"}."),
            .too_many_requests => try w.writer.writeAll("The search service is rate-limited right now. Answer from what you know and say that the search could not be completed."),
            .unavailable => try w.writer.writeAll("The web search service is unavailable. Answer from what you know and say that the search could not be performed."),
        },
    }
    return w.written();
}

/// The tool-result text the model read for a `web_search_tool_result` block
/// that comes back in conversation history (`input_json`: the input of its
/// `server_tool_use`): `modelText` of the same results, so a later turn
/// renders exactly what the search loop fed the model. Snippets are recovered
/// from our `encrypted_content`; another server's opaque payload leaves title
/// and URL only.
pub fn historyResultText(arena: Allocator, input_json: []const u8, content: std.json.Value) ![]const u8 {
    const query = queryOf(arena, input_json) orelse "";
    const outcome: Outcome = switch (content) {
        .array => |arr| blk: {
            var rs = std.ArrayList(Result).empty;
            for (arr.items) |item| {
                if (item != .object) continue;
                const url = strField(item.object, "url") orelse continue;
                var snippet: []const u8 = "";
                if (strField(item.object, "encrypted_content")) |enc| {
                    if (std.mem.startsWith(u8, enc, ENCRYPTED_PREFIX)) snippet = decodeSnippet(arena, enc[ENCRYPTED_PREFIX.len..]);
                }
                try rs.append(arena, .{
                    .title = strField(item.object, "title") orelse url,
                    .url = url,
                    .snippet = snippet,
                    .page_age = strField(item.object, "page_age"),
                });
            }
            break :blk .{ .results = rs.items };
        },
        .object => |o| .{ .failed = std.meta.stringToEnum(ErrorCode, strField(o, "error_code") orelse "") orelse .unavailable },
        else => .{ .failed = .unavailable },
    };
    return modelText(arena, query, outcome);
}

fn decodeSnippet(arena: Allocator, b64: []const u8) []const u8 {
    const dec = std.base64.standard.Decoder;
    const n = dec.calcSizeForSlice(b64) catch return "";
    const buf = arena.alloc(u8, n) catch return "";
    dec.decode(buf, b64) catch return "";
    return buf;
}

/// The `content` of a `web_search_tool_result` block: the result array, or the error object.
pub fn resultContentJson(arena: Allocator, outcome: Outcome) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(arena);
    switch (outcome) {
        .failed => |code| try w.writer.print("{{\"type\":\"web_search_tool_result_error\",\"error_code\":\"{s}\"}}", .{@tagName(code)}),
        .results => |rs| {
            try w.writer.writeByte('[');
            for (rs, 0..) |r, i| {
                if (i > 0) try w.writer.writeByte(',');
                try w.writer.writeAll("{\"type\":\"web_search_result\",\"title\":");
                try writeJsonString(&w.writer, r.title);
                try w.writer.writeAll(",\"url\":");
                try writeJsonString(&w.writer, r.url);
                try w.writer.writeAll(",\"encrypted_content\":\"" ++ ENCRYPTED_PREFIX);
                const enc = std.base64.standard.Encoder;
                const b64 = try arena.alloc(u8, enc.calcSize(r.snippet.len));
                try w.writer.writeAll(enc.encode(b64, r.snippet));
                try w.writer.writeAll("\",\"page_age\":");
                if (r.page_age) |age| try writeJsonString(&w.writer, age) else try w.writer.writeAll("null");
                try w.writer.writeByte('}');
            }
            try w.writer.writeByte(']');
        },
    }
    return w.written();
}

pub fn resultBlockJson(arena: Allocator, tool_use_id: []const u8, content_json: []const u8) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(arena);
    try w.writer.writeAll("{\"type\":\"web_search_tool_result\",\"tool_use_id\":");
    try writeJsonString(&w.writer, tool_use_id);
    try w.writer.print(",\"content\":{s}}}", .{content_json});
    return w.written();
}

pub fn serverToolUseJson(arena: Allocator, id: []const u8, name: []const u8, input_json: []const u8) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(arena);
    try w.writer.writeAll("{\"type\":\"server_tool_use\",\"id\":");
    try writeJsonString(&w.writer, id);
    try w.writer.writeAll(",\"name\":");
    try writeJsonString(&w.writer, name);
    try w.writer.print(",\"input\":{s}}}", .{input_json});
    return w.written();
}

/// `srvtoolu_…` id for the server tool call the model made as `inner_id`.
pub fn serverToolId(arena: Allocator, inner_id: []const u8) ![]const u8 {
    const suffix = if (std.mem.startsWith(u8, inner_id, "toolu_")) inner_id["toolu_".len..] else inner_id;
    return std.fmt.allocPrint(arena, "srvtoolu_{s}", .{suffix});
}

/// The `query` of a web_search call's input JSON, or null.
pub fn queryOf(arena: Allocator, input_json: []const u8) ?[]const u8 {
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena, input_json, .{}) catch return null;
    if (v != .object) return null;
    const q = strField(v.object, "query") orelse return null;
    const t = std.mem.trim(u8, q, " \t\r\n");
    return if (t.len > 0) t else null;
}

/// `input_json` when it is a JSON object, else "{}" (a half-written call must not corrupt the next body).
pub fn objectOrEmpty(arena: Allocator, input_json: []const u8) []const u8 {
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena, input_json, .{}) catch return "{}";
    return if (v == .object) input_json else "{}";
}

pub const Usage = struct {
    input_tokens: u64 = 0,
    output_tokens: u64 = 0,
    cache_read: u64 = 0,
    searches: u32 = 0,

    fn write(u: Usage, w: *std.Io.Writer) !void {
        try w.print("{{\"input_tokens\":{d},\"output_tokens\":{d},\"cache_read_input_tokens\":{d},\"server_tool_use\":{{\"web_search_requests\":{d}}}}}", .{
            u.input_tokens, u.output_tokens, u.cache_read, u.searches,
        });
    }
};

/// Non-streaming response body.
pub fn messageJson(arena: Allocator, msg_id: []const u8, model: []const u8, blocks: []const []const u8, stop_reason: []const u8, stop_sequence_json: []const u8, usage: Usage) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(arena);
    try w.writer.writeAll("{\"id\":");
    try writeJsonString(&w.writer, msg_id);
    try w.writer.writeAll(",\"type\":\"message\",\"role\":\"assistant\",\"content\":[");
    for (blocks, 0..) |b, i| {
        if (i > 0) try w.writer.writeByte(',');
        try w.writer.writeAll(b);
    }
    try w.writer.writeAll("],\"model\":");
    try writeJsonString(&w.writer, model);
    try w.writer.writeAll(",\"stop_reason\":");
    try writeJsonString(&w.writer, stop_reason);
    try w.writer.print(",\"stop_sequence\":{s},\"usage\":", .{stop_sequence_json});
    try usage.write(&w.writer);
    try w.writer.writeByte('}');
    return w.written();
}

// ── Capture of the inner /v1/messages stream ────────────────────────────────

pub const BlockKind = enum { text, thinking, tool_use, other };

pub const InnerBlock = struct {
    kind: BlockKind = .other,
    id: []const u8 = "",
    name: []const u8 = "",
    /// text / thinking / the tool call's input JSON, accumulated from the deltas.
    body: std.ArrayList(u8) = .empty,
    /// Index this block was relayed under, when it was relayed.
    outer_index: ?u32 = null,
    /// A call of the web_search function (answered here, never relayed as tool_use).
    is_search: bool = false,

    /// The block as an Anthropic content block, or null when it carries nothing.
    pub fn json(b: *const InnerBlock, arena: Allocator) !?[]const u8 {
        var w: std.Io.Writer.Allocating = .init(arena);
        switch (b.kind) {
            .text => {
                if (b.body.items.len == 0) return null;
                try w.writer.writeAll("{\"type\":\"text\",\"text\":");
                try writeJsonString(&w.writer, b.body.items);
                try w.writer.writeByte('}');
            },
            .thinking => {
                if (b.body.items.len == 0) return null;
                try w.writer.writeAll("{\"type\":\"thinking\",\"thinking\":");
                try writeJsonString(&w.writer, b.body.items);
                try w.writer.writeAll(",\"signature\":\"" ++ SIGNATURE ++ "\"}");
            },
            .tool_use => {
                try w.writer.writeAll("{\"type\":\"tool_use\",\"id\":");
                try writeJsonString(&w.writer, b.id);
                try w.writer.writeAll(",\"name\":");
                try writeJsonString(&w.writer, b.name);
                try w.writer.print(",\"input\":{s}}}", .{objectOrEmpty(arena, b.body.items)});
            },
            .other => return null,
        }
        return w.written();
    }
};

/// Receives every byte one inner /v1/messages call writes (HTTP head, then
/// SSE events — or a JSON error body when the handler refused the request).
pub const Capture = struct {
    /// Transient per-event JSON parses.
    gpa: Allocator,
    /// Everything that outlives an event (blocks, head, error body).
    arena: Allocator,
    /// The web_search function name; calls to it are answered by the loop.
    search_name: []const u8,
    relay: ?*Relay = null,

    buf: std.ArrayList(u8) = .empty,
    head_done: bool = false,
    /// The raw response head through the blank line.
    head: []const u8 = "",
    status: u16 = 0,
    is_sse: bool = false,
    /// A non-SSE response body (the handler's JSON error).
    body: std.ArrayList(u8) = .empty,
    blocks: std.ArrayList(InnerBlock) = .empty,
    stop_reason: []const u8 = "end_turn",
    stop_sequence_json: []const u8 = "null",
    input_tokens: u64 = 0,
    output_tokens: u64 = 0,
    cache_read: u64 = 0,
    /// Data of an SSE `error` event.
    error_data: ?[]const u8 = null,
    saw_stop: bool = false,

    pub fn feed(self: *Capture, bytes: []const u8) anyerror!void {
        if (self.head_done and !self.is_sse) return self.body.appendSlice(self.arena, bytes);
        try self.buf.appendSlice(self.arena, bytes);
        if (!self.head_done) {
            const he = std.mem.indexOf(u8, self.buf.items, "\r\n\r\n") orelse return;
            self.head = try self.arena.dupe(u8, self.buf.items[0 .. he + 4]);
            if (std.mem.startsWith(u8, self.head, "HTTP/1.") and self.head.len >= 12) {
                self.status = std.fmt.parseInt(u16, self.head[9..12], 10) catch 500;
            } else self.status = 500;
            self.is_sse = std.ascii.findIgnoreCase(self.head, "text/event-stream") != null;
            self.consume(he + 4);
            self.head_done = true;
            if (!self.is_sse) {
                try self.body.appendSlice(self.arena, self.buf.items);
                self.buf.clearRetainingCapacity();
                return;
            }
        }
        while (std.mem.indexOf(u8, self.buf.items, "\n\n")) |ee| {
            const ev = try self.gpa.dupe(u8, self.buf.items[0..ee]);
            defer self.gpa.free(ev);
            self.consume(ee + 2);
            try self.handleEvent(ev);
        }
    }

    fn consume(self: *Capture, n: usize) void {
        const rest = self.buf.items.len - n;
        std.mem.copyForwards(u8, self.buf.items[0..rest], self.buf.items[n..]);
        self.buf.items.len = rest;
    }

    fn handleEvent(self: *Capture, ev: []const u8) !void {
        var name: []const u8 = "";
        var data: []const u8 = "";
        var lines = std.mem.splitScalar(u8, ev, '\n');
        while (lines.next()) |raw_line| {
            const line = std.mem.trimEnd(u8, raw_line, "\r");
            if (std.mem.startsWith(u8, line, "event:")) {
                name = std.mem.trim(u8, line["event:".len..], " ");
            } else if (std.mem.startsWith(u8, line, "data:")) {
                data = std.mem.trimStart(u8, line["data:".len..], " ");
            }
        }
        if (data.len == 0) return; // SSE comment / keepalive
        const parsed = std.json.parseFromSlice(std.json.Value, self.gpa, data, .{}) catch return;
        defer parsed.deinit();
        if (parsed.value != .object) return;
        const obj = parsed.value.object;
        const typ = if (name.len > 0) name else (strField(obj, "type") orelse return);

        if (std.mem.eql(u8, typ, "message_start")) {
            if (obj.get("message")) |m| if (m == .object) if (m.object.get("usage")) |u| if (u == .object) {
                self.input_tokens = intField(u.object, "input_tokens");
            };
        } else if (std.mem.eql(u8, typ, "content_block_start")) {
            const idx = intField(obj, "index");
            if (idx > 4096) return;
            while (self.blocks.items.len <= idx) try self.blocks.append(self.arena, .{});
            const b = &self.blocks.items[idx];
            b.* = .{};
            if (obj.get("content_block")) |cb| if (cb == .object) {
                const ct = strField(cb.object, "type") orelse "";
                if (std.mem.eql(u8, ct, "text")) {
                    b.kind = .text;
                    try b.body.appendSlice(self.arena, strField(cb.object, "text") orelse "");
                } else if (std.mem.eql(u8, ct, "thinking")) {
                    b.kind = .thinking;
                    try b.body.appendSlice(self.arena, strField(cb.object, "thinking") orelse "");
                } else if (std.mem.eql(u8, ct, "tool_use")) {
                    b.kind = .tool_use;
                    b.id = try self.arena.dupe(u8, strField(cb.object, "id") orelse "");
                    b.name = try self.arena.dupe(u8, strField(cb.object, "name") orelse "");
                    b.is_search = std.mem.eql(u8, b.name, self.search_name);
                }
            };
        } else if (std.mem.eql(u8, typ, "content_block_delta")) {
            const idx = intField(obj, "index");
            if (idx >= self.blocks.items.len) return;
            const b = &self.blocks.items[idx];
            const d = obj.get("delta") orelse return;
            if (d != .object) return;
            const dt = strField(d.object, "type") orelse "";
            const piece = if (std.mem.eql(u8, dt, "text_delta"))
                strField(d.object, "text")
            else if (std.mem.eql(u8, dt, "thinking_delta"))
                strField(d.object, "thinking")
            else if (std.mem.eql(u8, dt, "input_json_delta"))
                strField(d.object, "partial_json")
            else
                null;
            if (piece) |p| try b.body.appendSlice(self.arena, p);
        } else if (std.mem.eql(u8, typ, "message_delta")) {
            if (obj.get("delta")) |d| if (d == .object) {
                if (strField(d.object, "stop_reason")) |sr| self.stop_reason = try self.arena.dupe(u8, sr);
                if (d.object.get("stop_sequence")) |ss| if (ss == .string) {
                    var w: std.Io.Writer.Allocating = .init(self.arena);
                    try writeJsonString(&w.writer, ss.string);
                    self.stop_sequence_json = w.written();
                };
            };
            if (obj.get("usage")) |u| if (u == .object) {
                self.output_tokens = intField(u.object, "output_tokens");
                self.cache_read = intField(u.object, "cache_read_input_tokens");
            };
        } else if (std.mem.eql(u8, typ, "message_stop")) {
            self.saw_stop = true;
        } else if (std.mem.eql(u8, typ, "error")) {
            self.error_data = try self.arena.dupe(u8, data);
        }
        if (self.relay) |r| try r.onInner(self, typ, data);
    }

    pub fn searchCount(self: *const Capture) usize {
        var n: usize = 0;
        for (self.blocks.items) |b| if (b.kind == .tool_use and b.is_search) {
            n += 1;
        };
        return n;
    }

    pub fn hasClientToolCall(self: *const Capture) bool {
        for (self.blocks.items) |b| if (b.kind == .tool_use and !b.is_search) return true;
        return false;
    }
};

fn intField(o: std.json.ObjectMap, key: []const u8) u64 {
    const v = o.get(key) orelse return 0;
    return if (v == .integer and v.integer >= 0) @intCast(v.integer) else 0;
}

/// `data` with its first `"index":N` replaced by `new_index`.
pub fn rewriteIndex(allocator: Allocator, data: []const u8, new_index: u32) ![]u8 {
    const key = "\"index\":";
    const at = std.mem.indexOf(u8, data, key) orelse return allocator.dupe(u8, data);
    var end = at + key.len;
    while (end < data.len and std.ascii.isDigit(data[end])) end += 1;
    return std.fmt.allocPrint(allocator, "{s}{s}{d}{s}", .{ data[0..at], key, new_index, data[end..] });
}

/// The outer SSE stream of a web-search request. Started lazily on the first
/// inner `message_start`, so a request the handler refuses outright still
/// gets its plain HTTP error.
pub const Relay = struct {
    gpa: Allocator,
    out_ctx: *anyopaque,
    /// Writes to the client socket, bypassing the Conn capture hook.
    outFn: *const fn (ctx: *anyopaque, data: []const u8) anyerror!void,
    /// The SSE response head, written once at start.
    sse_head: []const u8,
    model: []const u8,
    msg_id: []const u8,
    started: bool = false,
    next_index: u32 = 0,

    pub fn event(self: *Relay, name: []const u8, data: []const u8) !void {
        const frame = try std.fmt.allocPrint(self.gpa, "event: {s}\ndata: {s}\n\n", .{ name, data });
        defer self.gpa.free(frame);
        try self.outFn(self.out_ctx, frame);
    }

    pub fn start(self: *Relay, input_tokens: u64) !void {
        if (self.started) return;
        try self.outFn(self.out_ctx, self.sse_head);
        self.started = true;
        var w: std.Io.Writer.Allocating = .init(self.gpa);
        defer w.deinit();
        try w.writer.writeAll("{\"type\":\"message_start\",\"message\":{\"id\":");
        try writeJsonString(&w.writer, self.msg_id);
        try w.writer.writeAll(",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":");
        try writeJsonString(&w.writer, self.model);
        try w.writer.print(",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{{\"input_tokens\":{d},\"output_tokens\":1}}}}}}", .{input_tokens});
        try self.event("message_start", w.written());
        try self.event("ping", PING);
    }

    pub fn ping(self: *Relay) !void {
        if (self.started) try self.event("ping", PING);
    }

    fn onInner(self: *Relay, cap: *Capture, typ: []const u8, data: []const u8) !void {
        if (std.mem.eql(u8, typ, "message_start")) return self.start(cap.input_tokens);
        if (std.mem.eql(u8, typ, "ping")) return self.ping();
        if (std.mem.eql(u8, typ, "error")) {
            if (self.started) try self.event("error", data);
            return;
        }
        const is_start = std.mem.eql(u8, typ, "content_block_start");
        if (!is_start and !std.mem.eql(u8, typ, "content_block_delta") and !std.mem.eql(u8, typ, "content_block_stop")) return;
        if (!self.started) return;
        const idx = indexOfEvent(data) orelse return;
        if (idx >= cap.blocks.items.len) return;
        const b = &cap.blocks.items[idx];
        if (b.is_search) return;
        if (is_start) {
            b.outer_index = self.next_index;
            self.next_index += 1;
        }
        const oi = b.outer_index orelse return;
        const rewritten = try rewriteIndex(self.gpa, data, oi);
        defer self.gpa.free(rewritten);
        try self.event(typ, rewritten);
    }

    /// server_tool_use block for one search: start, the whole input as one delta, stop.
    pub fn serverToolUse(self: *Relay, id: []const u8, name: []const u8, input_json: []const u8) !void {
        const i = self.next_index;
        self.next_index += 1;
        var w: std.Io.Writer.Allocating = .init(self.gpa);
        defer w.deinit();
        try w.writer.print("{{\"type\":\"content_block_start\",\"index\":{d},\"content_block\":{{\"type\":\"server_tool_use\",\"id\":", .{i});
        try writeJsonString(&w.writer, id);
        try w.writer.writeAll(",\"name\":");
        try writeJsonString(&w.writer, name);
        try w.writer.writeAll(",\"input\":{}}}");
        try self.event("content_block_start", w.written());
        w.clearRetainingCapacity();
        try w.writer.print("{{\"type\":\"content_block_delta\",\"index\":{d},\"delta\":{{\"type\":\"input_json_delta\",\"partial_json\":", .{i});
        try writeJsonString(&w.writer, input_json);
        try w.writer.writeAll("}}");
        try self.event("content_block_delta", w.written());
        try self.stopBlock(i);
    }

    /// web_search_tool_result block: the whole content in the start event, then stop.
    pub fn searchResult(self: *Relay, tool_use_id: []const u8, content_json: []const u8) !void {
        const i = self.next_index;
        self.next_index += 1;
        var w: std.Io.Writer.Allocating = .init(self.gpa);
        defer w.deinit();
        try w.writer.print("{{\"type\":\"content_block_start\",\"index\":{d},\"content_block\":{{\"type\":\"web_search_tool_result\",\"tool_use_id\":", .{i});
        try writeJsonString(&w.writer, tool_use_id);
        try w.writer.print(",\"content\":{s}}}}}", .{content_json});
        try self.event("content_block_start", w.written());
        try self.stopBlock(i);
    }

    fn stopBlock(self: *Relay, i: u32) !void {
        var buf: [64]u8 = undefined;
        try self.event("content_block_stop", try std.fmt.bufPrint(&buf, "{{\"type\":\"content_block_stop\",\"index\":{d}}}", .{i}));
    }

    pub fn finish(self: *Relay, stop_reason: []const u8, stop_sequence_json: []const u8, usage: Usage) !void {
        var w: std.Io.Writer.Allocating = .init(self.gpa);
        defer w.deinit();
        try w.writer.writeAll("{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":");
        try writeJsonString(&w.writer, stop_reason);
        try w.writer.print(",\"stop_sequence\":{s}}},\"usage\":", .{stop_sequence_json});
        try usage.write(&w.writer);
        try w.writer.writeByte('}');
        try self.event("message_delta", w.written());
        try self.event("message_stop", "{\"type\":\"message_stop\"}");
    }
};

/// The `index` of a content_block_* event payload.
fn indexOfEvent(data: []const u8) ?usize {
    const key = "\"index\":";
    const at = std.mem.indexOf(u8, data, key) orelse return null;
    var end = at + key.len;
    while (end < data.len and std.ascii.isDigit(data[end])) end += 1;
    return std.fmt.parseInt(usize, data[at + key.len .. end], 10) catch null;
}

// ── Inner request bodies ────────────────────────────────────────────────────

/// One answered call: the tool_result text the model reads next round.
pub const ToolAnswer = struct { tool_use_id: []const u8, text: []const u8 };

/// The assistant turn the model just produced (its blocks, search calls
/// included) plus the user turn carrying the search results, as two JSON
/// messages for the next round's `messages`.
pub fn roundMessages(arena: Allocator, cap: *const Capture, answers: []const ToolAnswer) ![2][]const u8 {
    var a: std.Io.Writer.Allocating = .init(arena);
    try a.writer.writeAll("{\"role\":\"assistant\",\"content\":[");
    var n: usize = 0;
    for (cap.blocks.items) |*b| {
        const j = (try b.json(arena)) orelse continue;
        if (n > 0) try a.writer.writeByte(',');
        try a.writer.writeAll(j);
        n += 1;
    }
    try a.writer.writeAll("]}");
    var u: std.Io.Writer.Allocating = .init(arena);
    try u.writer.writeAll("{\"role\":\"user\",\"content\":[");
    for (answers, 0..) |ans, i| {
        if (i > 0) try u.writer.writeByte(',');
        try u.writer.writeAll("{\"type\":\"tool_result\",\"tool_use_id\":");
        try writeJsonString(&u.writer, ans.tool_use_id);
        try u.writer.writeAll(",\"content\":");
        try writeJsonString(&u.writer, ans.text);
        try u.writer.writeByte('}');
    }
    try u.writer.writeAll("]}");
    return .{ a.written(), u.written() };
}

/// The request body for one inner round: the client's body with `stream`
/// forced on and `extra_messages` appended to `messages`. `relax_tool_choice`
/// (rounds after the first) turns a forcing tool_choice (`any` / `tool`) back
/// to auto, or the model could never stop searching.
pub fn innerBody(arena: Allocator, root: std.json.ObjectMap, extra_messages: []const []const u8, relax_tool_choice: bool) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(arena);
    try w.writer.writeByte('{');
    var it = root.iterator();
    while (it.next()) |kv| {
        const k = kv.key_ptr.*;
        if (std.mem.eql(u8, k, "stream") or std.mem.eql(u8, k, "messages")) continue;
        if (relax_tool_choice and std.mem.eql(u8, k, "tool_choice") and forcingToolChoice(kv.value_ptr.*)) continue;
        try writeJsonString(&w.writer, k);
        try w.writer.writeByte(':');
        try std.json.Stringify.value(kv.value_ptr.*, .{}, &w.writer);
        try w.writer.writeByte(',');
    }
    try w.writer.writeAll("\"stream\":true,\"messages\":[");
    var n: usize = 0;
    if (root.get("messages")) |mv| if (mv == .array) {
        for (mv.array.items) |m| {
            if (n > 0) try w.writer.writeByte(',');
            try std.json.Stringify.value(m, .{}, &w.writer);
            n += 1;
        }
    };
    for (extra_messages) |m| {
        if (n > 0) try w.writer.writeByte(',');
        try w.writer.writeAll(m);
        n += 1;
    }
    try w.writer.writeAll("]}");
    return w.written();
}

fn forcingToolChoice(v: std.json.Value) bool {
    if (v != .object) return false;
    const t = strField(v.object, "type") orelse return false;
    return std.mem.eql(u8, t, "any") or std.mem.eql(u8, t, "tool");
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

const testing = std.testing;

test "websearch: parseEndpoint takes http ipv4/localhost with port and prefix" {
    const a = parseEndpoint("http://127.0.0.1:8888").?;
    try testing.expectEqual([4]u8{ 127, 0, 0, 1 }, a.ip4);
    try testing.expectEqual(@as(u16, 8888), a.port);
    try testing.expectEqualStrings("", a.base_path);
    const b = parseEndpoint("http://localhost/searxng/").?;
    try testing.expectEqual([4]u8{ 127, 0, 0, 1 }, b.ip4);
    try testing.expectEqual(@as(u16, 80), b.port);
    try testing.expectEqualStrings("/searxng", b.base_path);
    try testing.expect(parseEndpoint("https://127.0.0.1:8888") == null);
    try testing.expect(parseEndpoint("http://search.example:8888") == null);
    try testing.expect(parseEndpoint("http://1.2.3:80") == null);
    try testing.expect(parseEndpoint("http://1.2.3.4:99999") == null);
}

test "websearch: searchPath form-encodes the query" {
    const ep = parseEndpoint("http://127.0.0.1:8888/s").?;
    const p = try searchPath(testing.allocator, ep, "zig 0.15 & \"async\"/io");
    defer testing.allocator.free(p);
    try testing.expectEqualStrings("/s/search?q=zig+0.15+%26+%22async%22%2Fio&format=json", p);
}

test "websearch: findServerTool reads name, max_uses and domain lists" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const body =
        \\[{"name":"Read","input_schema":{}},{"type":"web_search_20250305","name":"web_search","max_uses":3,"allowed_domains":["docs.python.org"],"blocked_domains":["x.com"]}]
    ;
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{});
    const cfg = (try findServerTool(arena, v.array.items)).?;
    try testing.expectEqualStrings("web_search", cfg.name);
    try testing.expectEqual(@as(u32, 3), cfg.max_uses);
    try testing.expectEqual(@as(usize, 1), cfg.allowed_domains.len);
    try testing.expectEqualStrings("x.com", cfg.blocked_domains[0]);
    const none = try std.json.parseFromSliceLeaky(std.json.Value, arena, "[{\"name\":\"Bash\"}]", .{});
    try testing.expect((try findServerTool(arena, none.array.items)) == null);
}

test "websearch: domain filters match subdomains, not look-alikes" {
    const cfg = ToolConfig{ .name = "web_search", .allowed_domains = &.{ "https://www.python.org/about", "github.com" }, .blocked_domains = &.{"gist.github.com"} };
    try testing.expect(domainAllowed("https://docs.python.org/3/", cfg));
    try testing.expect(domainAllowed("https://python.org", cfg));
    try testing.expect(domainAllowed("https://github.com/ml-explore/mlx", cfg));
    try testing.expect(!domainAllowed("https://gist.github.com/x", cfg));
    try testing.expect(!domainAllowed("https://notgithub.com/x", cfg));
    try testing.expect(!domainAllowed("https://example.com/", cfg));
    try testing.expect(domainAllowed("https://example.com/", .{ .name = "web_search" }));
}

test "websearch: parseSearxResults filters, dedupes, trims and caps" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const body =
        \\{"query":"q","results":[
        \\ {"url":"https://a.example/1","title":"  A\n title ","content":"one\n two","publishedDate":"2026-09-01T10:00:00"},
        \\ {"url":"https://a.example/1","title":"dup","content":"dup"},
        \\ {"url":"ftp://b.example/","title":"ftp","content":""},
        \\ {"url":"https://blocked.example/x","title":"blocked","content":"no"},
        \\ {"url":"https://c.example/","content":"no title","publishedDate":null},
        \\ {"url":"https://d.example/","title":"D","content":"d"}
        \\]}
    ;
    const rs = try parseSearxResults(arena, body, .{ .name = "web_search", .blocked_domains = &.{"blocked.example"} }, 2);
    try testing.expectEqual(@as(usize, 2), rs.len);
    try testing.expectEqualStrings("A title", rs[0].title);
    try testing.expectEqualStrings("one two", rs[0].snippet);
    try testing.expectEqualStrings("2026-09-01", rs[0].page_age.?);
    try testing.expectEqualStrings("https://c.example/", rs[1].title);
    try testing.expect(rs[1].page_age == null);
    try testing.expectError(error.NoResults, parseSearxResults(arena, "{\"error\":1}", .{ .name = "web_search" }, 8));
}

test "websearch: parseHttpResponse handles content-length and chunked bodies" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const r1 = try parseHttpResponse(arena, "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}xx");
    try testing.expectEqual(@as(u16, 200), r1.status);
    try testing.expectEqualStrings("{}", r1.body);
    const r2 = try parseHttpResponse(arena, "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\n{\"a\r\n4;x=1\r\n\":1}\r\n0\r\n\r\n");
    try testing.expectEqualStrings("{\"a\":1}", r2.body);
    const r3 = try parseHttpResponse(arena, "HTTP/1.0 429 Too Many Requests\r\n\r\n");
    try testing.expectEqual(@as(u16, 429), r3.status);
}

test "websearch: modelText and resultContentJson render results and errors" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const rs = [_]Result{.{ .title = "T \"q\"", .url = "https://x.example/", .snippet = "snip", .page_age = "2026-01-02" }};
    const txt = try modelText(arena, "the query", .{ .results = &rs });
    try testing.expect(std.mem.indexOf(u8, txt, "Web search results for \"the query\":") != null);
    try testing.expect(std.mem.indexOf(u8, txt, "[1] T \"q\"\nURL: https://x.example/\nPublished: 2026-01-02\nsnip\n") != null);
    const cj = try resultContentJson(arena, .{ .results = &rs });
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena, cj, .{});
    try testing.expectEqualStrings("web_search_result", v.array.items[0].object.get("type").?.string);
    try testing.expectEqualStrings("T \"q\"", v.array.items[0].object.get("title").?.string);
    const enc = v.array.items[0].object.get("encrypted_content").?.string;
    try testing.expect(std.mem.startsWith(u8, enc, ENCRYPTED_PREFIX));
    try testing.expectEqualStrings("{\"type\":\"web_search_tool_result_error\",\"error_code\":\"max_uses_exceeded\"}", try resultContentJson(arena, .{ .failed = .max_uses_exceeded }));
    try testing.expect(std.mem.indexOf(u8, try modelText(arena, "q", .{ .failed = .unavailable }), "unavailable") != null);
    try testing.expect(std.mem.indexOf(u8, try modelText(arena, "q", .{ .results = &.{} }), "No web results") != null);
}

const TestOut = struct {
    buf: std.ArrayList(u8) = .empty,
    fn write(ctx: *anyopaque, data: []const u8) anyerror!void {
        const self: *TestOut = @ptrCast(@alignCast(ctx));
        try self.buf.appendSlice(testing.allocator, data);
    }
};

const INNER_STREAM =
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n" ++
    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":42,\"output_tokens\":1}}}\n\n" ++
    "event: ping\ndata: {\"type\":\"ping\"}\n\n" ++
    "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n" ++
    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"need data\"}}\n\n" ++
    "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" ++
    "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" ++
    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Searching.\"}}\n\n" ++
    "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n" ++
    "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_7_0\",\"name\":\"web_search\",\"input\":{}}}\n\n" ++
    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"query\\\": \\\"mlx rel\"}}\n\n" ++
    "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"ease\\\"}\"}}\n\n" ++
    "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":2}\n\n" ++
    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":17,\"cache_read_input_tokens\":30}}\n\n" ++
    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

test "websearch: Capture accumulates an inner stream fed in arbitrary pieces" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var cap = Capture{ .gpa = testing.allocator, .arena = arena, .search_name = "web_search" };
    var i: usize = 0;
    var step: usize = 1;
    while (i < INNER_STREAM.len) : (step = step % 13 + 1) {
        const end = @min(INNER_STREAM.len, i + step);
        try cap.feed(INNER_STREAM[i..end]);
        i = end;
    }
    try testing.expectEqual(@as(u16, 200), cap.status);
    try testing.expect(cap.is_sse and cap.saw_stop);
    try testing.expectEqual(@as(u64, 42), cap.input_tokens);
    try testing.expectEqual(@as(u64, 17), cap.output_tokens);
    try testing.expectEqual(@as(u64, 30), cap.cache_read);
    try testing.expectEqualStrings("tool_use", cap.stop_reason);
    try testing.expectEqual(@as(usize, 3), cap.blocks.items.len);
    try testing.expectEqualStrings("need data", cap.blocks.items[0].body.items);
    try testing.expectEqualStrings("Searching.", cap.blocks.items[1].body.items);
    const call = cap.blocks.items[2];
    try testing.expect(call.is_search);
    try testing.expectEqualStrings("toolu_7_0", call.id);
    try testing.expectEqualStrings("mlx release", queryOf(arena, call.body.items).?);
    try testing.expectEqual(@as(usize, 1), cap.searchCount());
    try testing.expect(!cap.hasClientToolCall());

    const msgs = try roundMessages(arena, &cap, &.{.{ .tool_use_id = "toolu_7_0", .text = "results" }});
    const av = try std.json.parseFromSliceLeaky(std.json.Value, arena, msgs[0], .{});
    const content = av.object.get("content").?.array.items;
    try testing.expectEqual(@as(usize, 3), content.len);
    try testing.expectEqualStrings("thinking", content[0].object.get("type").?.string);
    try testing.expectEqualStrings("mlx release", content[2].object.get("input").?.object.get("query").?.string);
    const uv = try std.json.parseFromSliceLeaky(std.json.Value, arena, msgs[1], .{});
    try testing.expectEqualStrings("tool_result", uv.object.get("content").?.array.items[0].object.get("type").?.string);
}

test "websearch: a non-SSE inner answer is kept verbatim for relaying" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    var cap = Capture{ .gpa = testing.allocator, .arena = arena_state.allocator(), .search_name = "web_search" };
    try cap.feed("HTTP/1.1 400 Error\r\nContent-Type: application/json\r\nContent-Length: 9\r\n\r\n{\"type\":");
    try cap.feed("1}");
    try testing.expectEqual(@as(u16, 400), cap.status);
    try testing.expect(!cap.is_sse);
    try testing.expectEqualStrings("{\"type\":1}", cap.body.items);
    try testing.expect(std.mem.endsWith(u8, cap.head, "\r\n\r\n"));
}

test "websearch: Relay renumbers relayed blocks and hides search calls" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var out = TestOut{};
    defer out.buf.deinit(testing.allocator);
    var relay = Relay{ .gpa = testing.allocator, .out_ctx = &out, .outFn = TestOut.write, .sse_head = "HEAD\r\n\r\n", .model = "m", .msg_id = "msg_x" };
    // Round 1 relays thinking(0), text(1); the call is answered by the loop.
    var cap = Capture{ .gpa = testing.allocator, .arena = arena, .search_name = "web_search", .relay = &relay };
    try cap.feed(INNER_STREAM);
    try relay.serverToolUse("srvtoolu_7_0", "web_search", "{\"query\":\"mlx release\"}");
    try relay.searchResult("srvtoolu_7_0", "[]");
    // Round 2: its text block 0 continues at outer index 4.
    var cap2 = Capture{ .gpa = testing.allocator, .arena = arena, .search_name = "web_search", .relay = &relay };
    try cap2.feed("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n" ++
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":90}}}\n\n" ++
        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" ++
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Answer\"}}\n\n" ++
        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" ++
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":5}}\n\n" ++
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
    try relay.finish("end_turn", "null", .{ .input_tokens = 132, .output_tokens = 22, .searches = 1 });

    const s = out.buf.items;
    try testing.expect(std.mem.startsWith(u8, s, "HEAD\r\n\r\nevent: message_start\n"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, s, "event: message_start"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, s, "event: message_stop"));
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, s, "event: message_delta"));
    try testing.expect(std.mem.indexOf(u8, s, "\"tool_use\"") == null);
    try testing.expect(std.mem.indexOf(u8, s, "{\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Searching.\"}}") != null);
    try testing.expect(std.mem.indexOf(u8, s, "\"index\":2,\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"srvtoolu_7_0\"") != null);
    try testing.expect(std.mem.indexOf(u8, s, "\"partial_json\":\"{\\\"query\\\":\\\"mlx release\\\"}\"") != null);
    try testing.expect(std.mem.indexOf(u8, s, "\"index\":3,\"content_block\":{\"type\":\"web_search_tool_result\"") != null);
    try testing.expect(std.mem.indexOf(u8, s, "{\"type\":\"content_block_delta\",\"index\":4,\"delta\":{\"type\":\"text_delta\",\"text\":\"Answer\"}}") != null);
    try testing.expect(std.mem.indexOf(u8, s, "\"server_tool_use\":{\"web_search_requests\":1}") != null);
    // Every relayed data line is valid JSON.
    var lines = std.mem.splitScalar(u8, s, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const pv = try std.json.parseFromSlice(std.json.Value, testing.allocator, line["data: ".len..], .{});
        pv.deinit();
    }
}

test "websearch: innerBody forces streaming, appends rounds, relaxes a forcing tool_choice" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const req =
        \\{"model":"m","max_tokens":100,"stream":false,"tool_choice":{"type":"tool","name":"web_search"},"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search_20250305","name":"web_search"}]}
    ;
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, req, .{});
    const first = try innerBody(arena, root.object, &.{}, false);
    const v1 = try std.json.parseFromSliceLeaky(std.json.Value, arena, first, .{});
    try testing.expect(v1.object.get("stream").?.bool);
    try testing.expect(v1.object.get("tool_choice") != null);
    try testing.expectEqual(@as(usize, 1), v1.object.get("messages").?.array.items.len);
    try testing.expectEqual(@as(i64, 100), v1.object.get("max_tokens").?.integer);
    const later = try innerBody(arena, root.object, &.{ "{\"role\":\"assistant\",\"content\":[]}", "{\"role\":\"user\",\"content\":[]}" }, true);
    const v2 = try std.json.parseFromSliceLeaky(std.json.Value, arena, later, .{});
    try testing.expect(v2.object.get("tool_choice") == null);
    try testing.expectEqual(@as(usize, 3), v2.object.get("messages").?.array.items.len);
    try testing.expectEqualStrings("web_search_20250305", v2.object.get("tools").?.array.items[0].object.get("type").?.string);
}

test "websearch: historyResultText rebuilds the exact text the loop fed the model" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const rs = [_]Result{
        .{ .title = "Zig 0.15.1 \"notes\"", .url = "https://ziglang.org/", .snippet = "Released — août", .page_age = "2025-08-20" },
        .{ .title = "B", .url = "https://b.example/", .snippet = "" },
    };
    const outcome: Outcome = .{ .results = &rs };
    const expected = try modelText(arena, "zig 0.15", outcome);
    const block = try resultBlockJson(arena, "srvtoolu_1", try resultContentJson(arena, outcome));
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena, block, .{});
    const got = try historyResultText(arena, "{\"query\":\" zig 0.15 \"}", v.object.get("content").?);
    try testing.expectEqualStrings(expected, got);
    // Errors keep their code; an unknown code reads as unavailable.
    const err_v = try std.json.parseFromSliceLeaky(std.json.Value, arena, "{\"type\":\"web_search_tool_result_error\",\"error_code\":\"max_uses_exceeded\"}", .{});
    try testing.expectEqualStrings(try modelText(arena, "q", .{ .failed = .max_uses_exceeded }), try historyResultText(arena, "{\"query\":\"q\"}", err_v));
    // Another server's opaque encrypted_content: title and URL only.
    const foreign = try std.json.parseFromSliceLeaky(std.json.Value, arena, "[{\"type\":\"web_search_result\",\"title\":\"T\",\"url\":\"https://t.example/\",\"encrypted_content\":\"EqAbCd\"}]", .{});
    try testing.expect(std.mem.indexOf(u8, try historyResultText(arena, "{}", foreign), "[1] T\nURL: https://t.example/\n") != null);
}

test "websearch: rewriteIndex and serverToolId" {
    const r = try rewriteIndex(testing.allocator, "{\"type\":\"content_block_stop\",\"index\":12}", 3);
    defer testing.allocator.free(r);
    try testing.expectEqualStrings("{\"type\":\"content_block_stop\",\"index\":3}", r);
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    try testing.expectEqualStrings("srvtoolu_9_1", try serverToolId(arena_state.allocator(), "toolu_9_1"));
    try testing.expectEqualStrings("srvtoolu_abc", try serverToolId(arena_state.allocator(), "abc"));
}
