//! Persistent usage log behind `GET /usage` (src/html/usage.html).
//!
//! One JSON line per text-generation request in `<dir>/usage-YYYY-MM.jsonl` (UTC month):
//! who sent it (Claude Code session + device from `metadata.user_id`, else the client's own
//! session field, else an anonymous key), which conversation thread inside that session (a
//! hash of the system prompt + first user turn: the main agent, each subagent and each side
//! request type are separate threads), what the model did (tokens, cache reuse, timings,
//! stop reason, tool calls, web searches) and what it was given (tools, images, documents).
//! Never prompt or answer text, except a short thread title (the first user turn, system
//! reminders stripped) that `MLX_SERVE_USAGE_TITLES=0` turns off.
//!
//! The last `MLX_SERVE_USAGE_DAYS` (default 90) stay in memory for the summary and
//! per-session queries; the files themselves are never pruned.
const std = @import("std");
const Allocator = std.mem.Allocator;

extern "c" fn fflush(stream: ?*std.c.FILE) c_int;

pub const Role = enum {
    /// The client's top-level conversation (Claude Code's interactive agent).
    main,
    /// A delegated agent inside the same session (Claude Code's Task tool).
    subagent,
    /// A no-tools, non-streaming helper call (Claude Code's auto-mode classifier, titles).
    side,
    /// Anything without a recognizable agent client.
    other,
};

pub const MAX_TOOL_NAMES = 16;
pub const TITLE_MAX = 120;

pub const Record = struct {
    /// Request start, unix milliseconds.
    ts_ms: i64 = 0,
    /// Start to the last byte, as the dispatcher saw it.
    dur_ms: u32 = 0,
    /// "messages" | "chat" | "responses" | "completions".
    route: []const u8 = "",
    session: []const u8 = "",
    device: []const u8 = "",
    /// User-Agent.
    client: []const u8 = "",
    /// Remote address (no port).
    peer: []const u8 = "",
    thread: u64 = 0,
    role: Role = .other,
    title: []const u8 = "",
    model: []const u8 = "",
    stream: bool = false,
    msgs: u32 = 0,
    tools: u32 = 0,
    images: u32 = 0,
    docs: u32 = 0,
    /// Generations this request ran (a server-side web search runs several).
    rounds: u32 = 0,
    prompt: u32 = 0,
    cached: u32 = 0,
    gen: u32 = 0,
    prefill_ms: u32 = 0,
    decode_ms: u32 = 0,
    stop: []const u8 = "",
    tool_calls: []const []const u8 = &.{},
    web_searches: u32 = 0,
    status: u16 = 200,
    /// Error type; "" when the request succeeded.
    err: []const u8 = "",
};

// ── Request identity (pure; unit-tested) ─────────────────────────────────────

pub const Identity = struct {
    session: []const u8 = "",
    device: []const u8 = "",
    thread: u64 = 0,
    role: Role = .other,
    /// Borrowed from the request JSON or written into `title_buf`.
    title: []const u8 = "",
    model: []const u8 = "",
    stream: bool = false,
    msgs: u32 = 0,
    tools: u32 = 0,
    images: u32 = 0,
    docs: u32 = 0,
};

pub const ClaudeUser = struct { session: []const u8, device: []const u8 };

/// Claude Code's `metadata.user_id`: a JSON string `{"device_id","account_uuid","session_id"}`
/// (2.x), or the older `user_<device>_account_<uuid>_session_<uuid>`. Null for anything else.
pub fn claudeUser(arena: Allocator, user_id: []const u8) ?ClaudeUser {
    const s = std.mem.trim(u8, user_id, " \t");
    if (s.len > 0 and s[0] == '{') {
        const v = std.json.parseFromSliceLeaky(std.json.Value, arena, s, .{}) catch return null;
        if (v != .object) return null;
        const sid = v.object.get("session_id") orelse return null;
        if (sid != .string or sid.string.len == 0) return null;
        const dev: []const u8 = if (v.object.get("device_id")) |d| (if (d == .string) d.string else "") else "";
        return .{ .session = sid.string, .device = dev };
    }
    if (std.mem.startsWith(u8, s, "user_")) {
        const at = std.mem.indexOf(u8, s, "_session_") orelse return null;
        const dev_end = std.mem.indexOf(u8, s, "_account_") orelse at;
        return .{ .session = s[at + "_session_".len ..], .device = s["user_".len..dev_end] };
    }
    return null;
}

fn textOf(v: std.json.Value, h: ?*std.hash.Wyhash, out: ?*std.ArrayList(u8), arena: Allocator) void {
    switch (v) {
        .string => |t| {
            if (h) |hh| hh.update(t);
            if (out) |o| o.appendSlice(arena, t) catch {};
        },
        .array => |parts| for (parts.items) |p| {
            if (p != .object) continue;
            const ty = p.object.get("type") orelse continue;
            if (ty != .string) continue;
            if (!std.mem.eql(u8, ty.string, "text") and !std.mem.eql(u8, ty.string, "input_text")) continue;
            const t = p.object.get("text") orelse continue;
            if (t != .string) continue;
            if (h) |hh| hh.update(t.string);
            if (out) |o| {
                if (o.items.len > 0) o.append(arena, '\n') catch {};
                o.appendSlice(arena, t.string) catch {};
            }
        },
        else => {},
    }
}

/// The first user turn with `<system-reminder>` blocks removed and whitespace collapsed,
/// cut to `TITLE_MAX` bytes on a UTF-8 boundary.
pub fn titleFrom(buf: []u8, text: []const u8) []const u8 {
    var n: usize = 0;
    var i: usize = 0;
    var space = false;
    const open = "<system-reminder>";
    const close = "</system-reminder>";
    const cap = @min(buf.len, TITLE_MAX);
    while (i < text.len and n < cap) {
        if (std.mem.startsWith(u8, text[i..], open)) {
            const end = std.mem.indexOfPos(u8, text, i, close) orelse text.len;
            i = @min(text.len, end + close.len);
            space = n > 0;
            continue;
        }
        const c = text[i];
        i += 1;
        if (c == ' ' or c == '\t' or c == '\n' or c == '\r') {
            space = n > 0;
            continue;
        }
        if (space) {
            if (n + 1 >= cap) break;
            buf[n] = ' ';
            n += 1;
            space = false;
        }
        buf[n] = c;
        n += 1;
    }
    // Never end inside a multi-byte sequence.
    var end = n;
    while (end > 0 and (buf[end - 1] & 0xC0) == 0x80) end -= 1;
    if (end > 0 and buf[end - 1] >= 0xC0) {
        const lead = buf[end - 1];
        const need: usize = if (lead >= 0xF0) 4 else if (lead >= 0xE0) 3 else 2;
        if (n - (end - 1) < need) n = end - 1;
    }
    return buf[0..n];
}

fn countMedia(v: std.json.Value, images: *u32, docs: *u32, depth: u8) void {
    if (depth > 6) return;
    switch (v) {
        .array => |parts| for (parts.items) |p| countMedia(p, images, docs, depth + 1),
        .object => |o| {
            if (o.get("type")) |ty| if (ty == .string) {
                const t = ty.string;
                if (std.mem.eql(u8, t, "image") or std.mem.eql(u8, t, "image_url") or std.mem.eql(u8, t, "input_image")) images.* += 1;
                if (std.mem.eql(u8, t, "document") or std.mem.eql(u8, t, "file") or std.mem.eql(u8, t, "input_file")) docs.* += 1;
            };
            // tool_result blocks nest their own content array.
            if (o.get("content")) |c| if (c == .array) countMedia(c, images, docs, depth + 1);
        },
        else => {},
    }
}

fn roleOfMessage(m: std.json.Value) []const u8 {
    if (m != .object) return "";
    const r = m.object.get("role") orelse return if (m.object.get("type")) |t| (if (t == .string) t.string else "") else "";
    return if (r == .string) r.string else "";
}

/// Everything the usage record wants from a request body, for /v1/messages,
/// /v1/chat/completions and /v1/responses alike. `title_buf` holds the title.
pub fn identify(arena: Allocator, root: std.json.ObjectMap, title_buf: []u8, titles: bool) Identity {
    var id: Identity = .{};
    if (root.get("model")) |m| if (m == .string) {
        id.model = m.string;
    };
    if (root.get("stream")) |s| if (s == .bool) {
        id.stream = s.bool;
    };
    if (root.get("tools")) |t| if (t == .array) {
        id.tools = @intCast(t.array.items.len);
    };

    // Session: Claude Code's metadata.user_id > a plain user_id > OpenAI's own routing fields.
    var plain: []const u8 = "";
    if (root.get("metadata")) |md| if (md == .object) if (md.object.get("user_id")) |u| if (u == .string) {
        if (claudeUser(arena, u.string)) |cu| {
            id.session = cu.session;
            id.device = cu.device;
        } else plain = u.string;
    };
    if (id.session.len == 0) {
        if (plain.len > 0) {
            id.session = plain;
        } else if (root.get("prompt_cache_key")) |v| {
            if (v == .string) id.session = v.string;
        } else if (root.get("user")) |v| {
            if (v == .string) id.session = v.string;
        }
    }

    // Conversation: Anthropic `messages` + `system`, OpenAI chat `messages` (system first),
    // Responses `input` + `instructions`.
    const msgs_v: ?std.json.Value = root.get("messages") orelse root.get("input");
    var system_text = std.ArrayList(u8).empty;
    var h = std.hash.Wyhash.init(0x75736167);
    if (root.get("system")) |s| textOf(s, &h, &system_text, arena);
    if (root.get("instructions")) |s| textOf(s, &h, &system_text, arena);
    var first_user: ?std.json.Value = null;
    if (msgs_v) |mv| switch (mv) {
        .array => |items| {
            id.msgs = @intCast(items.items.len);
            for (items.items) |m| {
                const role = roleOfMessage(m);
                if (std.mem.eql(u8, role, "system") or std.mem.eql(u8, role, "developer")) {
                    // Only the leading ones belong to the thread: Claude Code adds system
                    // messages after the first user turn as the conversation goes on.
                    if (first_user == null) if (m.object.get("content")) |c| textOf(c, &h, &system_text, arena);
                    continue;
                }
                if (first_user == null and std.mem.eql(u8, role, "user")) first_user = m.object.get("content");
                countMedia(m, &id.images, &id.docs, 0);
            }
        },
        .string => |s| {
            id.msgs = 1;
            first_user = .{ .string = s };
        },
        else => {},
    };
    h.update("\x00");
    var first_text = std.ArrayList(u8).empty;
    if (first_user) |fu| textOf(fu, &h, &first_text, arena);
    id.thread = h.final();
    if (titles) id.title = titleFrom(title_buf, first_text.items);

    const sys = system_text.items;
    const agentic = id.device.len > 0; // a Claude Code session
    if (agentic and id.tools == 0 and !id.stream) {
        // Agents always stream with tools; a bare call is a helper (the auto-mode
        // classifier's prompt quotes the agent's own, markers included).
        id.role = .side;
    } else if (std.mem.indexOf(u8, sys, "You are an agent for Claude Code") != null) {
        id.role = .subagent;
    } else if (std.mem.indexOf(u8, sys, "interactive agent") != null or std.mem.indexOf(u8, sys, "You are Claude Code") != null) {
        id.role = .main;
    } else {
        id.role = if (agentic) .subagent else .other;
    }
    return id;
}

/// The admission group `--max-active-per-user` counts a request under
/// (`scheduler.SubmitParams.admit_user`): Claude Code's device id, else the
/// client's address. 0 for a Claude Code helper call (no tools, not streamed:
/// the auto-mode classifier, titles), which must never queue behind agents:
/// the classifier gives up after about a minute and auto mode stalls.
pub fn admitKey(root: std.json.ObjectMap, peer: []const u8) u64 {
    var h = std.hash.Wyhash.init(0x61646d69);
    if (root.get("metadata")) |md| if (md == .object) if (md.object.get("user_id")) |u| if (u == .string) {
        var buf: [4096]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buf);
        if (claudeUser(fba.allocator(), u.string)) |cu| {
            const streamed = if (root.get("stream")) |v| v == .bool and v.bool else false;
            const tools = if (root.get("tools")) |v| v == .array and v.array.items.len > 0 else false;
            if (!streamed and !tools) return 0;
            h.update("dev:");
            h.update(if (cu.device.len > 0) cu.device else cu.session);
            return h.final() | 1;
        }
    };
    h.update("peer:");
    h.update(peer);
    return h.final() | 1;
}

// ── JSON lines ───────────────────────────────────────────────────────────────

pub fn writeStr(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    var start: usize = 0;
    for (s, 0..) |c, i| {
        const esc: ?[]const u8 = switch (c) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            else => null,
        };
        if (esc == null and c >= 0x20) continue;
        try w.writeAll(s[start..i]);
        if (esc) |e| try w.writeAll(e) else try w.print("\\u{x:0>4}", .{c});
        start = i + 1;
    }
    try w.writeAll(s[start..]);
    try w.writeByte('"');
}

pub fn writeLine(w: *std.Io.Writer, r: *const Record) !void {
    try w.print("{{\"ts\":{d},\"dur\":{d},\"route\":", .{ r.ts_ms, r.dur_ms });
    try writeStr(w, r.route);
    try w.writeAll(",\"sess\":");
    try writeStr(w, r.session);
    try w.writeAll(",\"dev\":");
    try writeStr(w, r.device);
    try w.writeAll(",\"client\":");
    try writeStr(w, r.client);
    try w.writeAll(",\"peer\":");
    try writeStr(w, r.peer);
    try w.print(",\"thread\":\"{x:0>16}\",\"role\":\"{s}\",\"title\":", .{ r.thread, @tagName(r.role) });
    try writeStr(w, r.title);
    try w.writeAll(",\"model\":");
    try writeStr(w, r.model);
    try w.print(",\"stream\":{},\"msgs\":{d},\"tools\":{d},\"img\":{d},\"doc\":{d},\"rounds\":{d},\"prompt\":{d},\"cached\":{d},\"gen\":{d},\"prefill_ms\":{d},\"decode_ms\":{d},\"stop\":", .{
        r.stream, r.msgs, r.tools, r.images, r.docs, r.rounds, r.prompt, r.cached, r.gen, r.prefill_ms, r.decode_ms,
    });
    try writeStr(w, r.stop);
    try w.writeAll(",\"calls\":[");
    for (r.tool_calls, 0..) |name, i| {
        if (i > 0) try w.writeByte(',');
        try writeStr(w, name);
    }
    try w.print("],\"ws\":{d},\"status\":{d},\"err\":", .{ r.web_searches, r.status });
    try writeStr(w, r.err);
    try w.writeAll("}\n");
}

const Line = struct {
    ts: i64 = 0,
    dur: u32 = 0,
    route: []const u8 = "",
    sess: []const u8 = "",
    dev: []const u8 = "",
    client: []const u8 = "",
    peer: []const u8 = "",
    thread: []const u8 = "",
    role: []const u8 = "",
    title: []const u8 = "",
    model: []const u8 = "",
    stream: bool = false,
    msgs: u32 = 0,
    tools: u32 = 0,
    img: u32 = 0,
    doc: u32 = 0,
    rounds: u32 = 0,
    prompt: u32 = 0,
    cached: u32 = 0,
    gen: u32 = 0,
    prefill_ms: u32 = 0,
    decode_ms: u32 = 0,
    stop: []const u8 = "",
    calls: []const []const u8 = &.{},
    ws: u32 = 0,
    status: u16 = 200,
    err: []const u8 = "",
};

/// One JSON line back into a Record; strings borrow `arena`. Null for a line that does not parse.
pub fn parseLine(arena: Allocator, line: []const u8) ?Record {
    const l = std.json.parseFromSliceLeaky(Line, arena, line, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch return null;
    if (l.ts == 0) return null;
    return .{
        .ts_ms = l.ts,
        .dur_ms = l.dur,
        .route = l.route,
        .session = l.sess,
        .device = l.dev,
        .client = l.client,
        .peer = l.peer,
        .thread = std.fmt.parseInt(u64, l.thread, 16) catch 0,
        .role = std.meta.stringToEnum(Role, l.role) orelse .other,
        .title = l.title,
        .model = l.model,
        .stream = l.stream,
        .msgs = l.msgs,
        .tools = l.tools,
        .images = l.img,
        .docs = l.doc,
        .rounds = l.rounds,
        .prompt = l.prompt,
        .cached = l.cached,
        .gen = l.gen,
        .prefill_ms = l.prefill_ms,
        .decode_ms = l.decode_ms,
        .stop = l.stop,
        .tool_calls = l.calls,
        .web_searches = l.ws,
        .status = l.status,
        .err = l.err,
    };
}

// ── Store ────────────────────────────────────────────────────────────────────

fn monthOf(ms: i64) u32 {
    const secs: u64 = @intCast(@max(0, @divFloor(ms, 1000)));
    const yd = (std.time.epoch.EpochSeconds{ .secs = secs }).getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    return @as(u32, yd.year) * 100 + md.month.numeric();
}

fn prevMonth(m: u32) u32 {
    return if (m % 100 == 1) (m / 100 - 1) * 100 + 12 else m - 1;
}

pub const Store = struct {
    gpa: Allocator,
    /// Interned strings: every record's text fields point here, so a session id, a client
    /// string or a thread title costs its bytes once however many requests repeat it.
    strings: std.heap.ArenaAllocator,
    pool: std.StringHashMapUnmanaged([]const u8) = .empty,
    records: std.ArrayList(Record) = .empty,
    dir: []const u8,
    retain_ms: i64,
    mu: std.c.pthread_mutex_t = .{},
    file: ?*std.c.FILE = null,
    file_month: u32 = 0,
    write_failed_logged: bool = false,

    pub fn init(gpa: Allocator, dir: []const u8, retain_days: u32, now_ms: i64) !*Store {
        const self = try gpa.create(Store);
        errdefer gpa.destroy(self);
        self.* = .{
            .gpa = gpa,
            .strings = std.heap.ArenaAllocator.init(gpa),
            .dir = try gpa.dupe(u8, dir),
            .retain_ms = @as(i64, retain_days) * std.time.ms_per_day,
        };
        var dir_z: [std.fs.max_path_bytes]u8 = undefined;
        const dz = std.fmt.bufPrintSentinel(&dir_z, "{s}", .{dir}, 0) catch return error.PathTooLong;
        // mkdir -p: each parent, then the directory itself (EEXIST is fine).
        for (1..dz.len) |i| if (dz[i] == '/') {
            dir_z[i] = 0;
            _ = std.c.mkdir(@ptrCast(&dir_z), 0o755);
            dir_z[i] = '/';
        };
        _ = std.c.mkdir(dz.ptr, 0o755);
        try self.load(now_ms);
        return self;
    }

    pub fn deinit(self: *Store) void {
        if (self.file) |f| _ = std.c.fclose(f);
        self.records.deinit(self.gpa);
        self.pool.deinit(self.gpa);
        self.strings.deinit();
        self.gpa.free(self.dir);
        self.gpa.destroy(self);
    }

    fn intern(self: *Store, s: []const u8) ![]const u8 {
        if (s.len == 0) return "";
        if (self.pool.get(s)) |hit| return hit;
        const owned = try self.strings.allocator().dupe(u8, s);
        try self.pool.put(self.gpa, owned, owned);
        return owned;
    }

    fn internRecord(self: *Store, r: Record) !Record {
        var out = r;
        out.route = try self.intern(r.route);
        out.session = try self.intern(r.session);
        out.device = try self.intern(r.device);
        out.client = try self.intern(r.client);
        out.peer = try self.intern(r.peer);
        out.title = try self.intern(r.title);
        out.model = try self.intern(r.model);
        out.stop = try self.intern(r.stop);
        out.err = try self.intern(r.err);
        if (r.tool_calls.len > 0) {
            const names = try self.strings.allocator().alloc([]const u8, @min(r.tool_calls.len, MAX_TOOL_NAMES));
            for (names, 0..) |*n, i| n.* = try self.intern(r.tool_calls[i]);
            out.tool_calls = names;
        } else out.tool_calls = &.{};
        return out;
    }

    fn pathFor(self: *const Store, buf: []u8, month: u32) ![:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/usage-{d}-{d:0>2}.jsonl", .{ self.dir, month / 100, month % 100 }, 0);
    }

    fn load(self: *Store, now_ms: i64) !void {
        const from = now_ms - self.retain_ms;
        var months: [64]u32 = undefined;
        var n: usize = 0;
        var m = monthOf(now_ms);
        const first = monthOf(from);
        while (n < months.len) {
            months[n] = m;
            n += 1;
            if (m <= first) break;
            m = prevMonth(m);
        }
        var parse_arena = std.heap.ArenaAllocator.init(self.gpa);
        defer parse_arena.deinit();
        var i = n;
        while (i > 0) {
            i -= 1;
            var path_buf: [std.fs.max_path_bytes]u8 = undefined;
            const path = try self.pathFor(&path_buf, months[i]);
            const f = std.c.fopen(path.ptr, "rb") orelse continue;
            defer _ = std.c.fclose(f);
            var data = std.ArrayList(u8).empty;
            defer data.deinit(self.gpa);
            var chunk: [1 << 16]u8 = undefined;
            while (true) {
                const got = std.c.fread(&chunk, 1, chunk.len, f);
                if (got == 0) break;
                try data.appendSlice(self.gpa, chunk[0..got]);
            }
            var it = std.mem.splitScalar(u8, data.items, '\n');
            while (it.next()) |line| {
                if (line.len < 2) continue;
                _ = parse_arena.reset(.retain_capacity);
                const r = parseLine(parse_arena.allocator(), line) orelse continue;
                if (r.ts_ms < from) continue;
                try self.records.append(self.gpa, try self.internRecord(r));
            }
        }
        // Files are appended in completion order; queries want start order.
        std.mem.sort(Record, self.records.items, {}, struct {
            fn lt(_: void, a: Record, b: Record) bool {
                return a.ts_ms < b.ts_ms;
            }
        }.lt);
    }

    /// Keep `r` (strings are copied) and append its line to this month's file.
    pub fn append(self: *Store, r: Record) void {
        _ = std.c.pthread_mutex_lock(&self.mu);
        defer _ = std.c.pthread_mutex_unlock(&self.mu);
        defer self.writeLocked(&r);
        const items = self.records.items;
        const newest = if (items.len > 0) @max(items[items.len - 1].ts_ms, r.ts_ms) else r.ts_ms;
        const horizon = newest - self.retain_ms;
        if (r.ts_ms < horizon) return; // the file keeps it; the in-memory window does not
        const kept = self.internRecord(r) catch return;
        // Requests finish out of order; keep the list sorted by start.
        var at = items.len;
        while (at > 0 and items[at - 1].ts_ms > kept.ts_ms) at -= 1;
        self.records.insert(self.gpa, at, kept) catch return;
        // Drop what aged out, a day's worth at a time (the files keep it).
        if (self.records.items[0].ts_ms < horizon - std.time.ms_per_day) {
            var cut: usize = 0;
            while (cut < self.records.items.len and self.records.items[cut].ts_ms < horizon) cut += 1;
            self.records.replaceRange(self.gpa, 0, cut, &.{}) catch {};
        }
    }

    fn writeLocked(self: *Store, r: *const Record) void {
        const month = monthOf(r.ts_ms);
        if (self.file == null or self.file_month != month) {
            if (self.file) |f| _ = std.c.fclose(f);
            self.file = null;
            var path_buf: [std.fs.max_path_bytes]u8 = undefined;
            const path = self.pathFor(&path_buf, month) catch return;
            self.file = std.c.fopen(path.ptr, "ab");
            self.file_month = month;
            if (self.file == null) {
                if (!self.write_failed_logged) std.debug.print("[usage] cannot open {s}; usage stays in memory only\n", .{path});
                self.write_failed_logged = true;
                return;
            }
        }
        var buf: [8192]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        writeLine(&w, r) catch return; // a line past 8 KB (16 long tool names, a huge title) is skipped
        const line = w.buffered();
        _ = std.c.fwrite(line.ptr, 1, line.len, self.file.?);
        _ = fflush(self.file);
    }

    pub fn count(self: *Store) usize {
        _ = std.c.pthread_mutex_lock(&self.mu);
        defer _ = std.c.pthread_mutex_unlock(&self.mu);
        return self.records.items.len;
    }

    // ── Queries ──────────────────────────────────────────────────────────────

    const SessAgg = struct {
        id: []const u8,
        title: []const u8 = "",
        title_main: bool = false,
        client: []const u8 = "",
        device: []const u8 = "",
        peer: []const u8 = "",
        first: i64 = std.math.maxInt(i64),
        last: i64 = 0,
        last_end: i64 = 0,
        requests: u32 = 0,
        roles: [4]u32 = .{ 0, 0, 0, 0 },
        threads: u32 = 0,
        prompt: u64 = 0,
        cached: u64 = 0,
        gen: u64 = 0,
        prefill_ms: u64 = 0,
        decode_ms: u64 = 0,
        busy_ms: u64 = 0,
        calls: u32 = 0,
        ws: u32 = 0,
        images: u32 = 0,
        docs: u32 = 0,
        errors: u32 = 0,
    };

    const Totals = struct {
        requests: u64 = 0,
        prompt: u64 = 0,
        cached: u64 = 0,
        gen: u64 = 0,
        prefill_ms: u64 = 0,
        decode_ms: u64 = 0,
        calls: u64 = 0,
        ws: u64 = 0,
        images: u64 = 0,
        docs: u64 = 0,
        errors: u64 = 0,

        fn add(t: *Totals, r: *const Record) void {
            t.requests += 1;
            t.prompt += r.prompt;
            t.cached += r.cached;
            t.gen += r.gen;
            t.prefill_ms += r.prefill_ms;
            t.decode_ms += r.decode_ms;
            t.calls += r.tool_calls.len;
            t.ws += r.web_searches;
            t.images += r.images;
            t.docs += r.docs;
            if (r.err.len > 0) t.errors += 1;
        }

        fn write(t: *const Totals, w: *std.Io.Writer) !void {
            try w.print("{{\"requests\":{d},\"prompt\":{d},\"cached\":{d},\"gen\":{d},\"prefill_ms\":{d},\"decode_ms\":{d},\"calls\":{d},\"ws\":{d},\"images\":{d},\"docs\":{d},\"errors\":{d}}}", .{
                t.requests, t.prompt, t.cached, t.gen, t.prefill_ms, t.decode_ms, t.calls, t.ws, t.images, t.docs, t.errors,
            });
        }
    };

    fn sessionKey(r: *const Record) []const u8 {
        return if (r.session.len > 0) r.session else r.client;
    }

    fn writeCounts(w: *std.Io.Writer, name: []const u8, map: *std.StringArrayHashMapUnmanaged(u64), limit: usize) !void {
        const Ctx = struct {
            vals: []const u64,
            pub fn lessThan(c: @This(), a: usize, b: usize) bool {
                return c.vals[a] > c.vals[b];
            }
        };
        map.sort(Ctx{ .vals = map.values() });
        try w.print("\"{s}\":[", .{name});
        for (map.keys(), map.values(), 0..) |k, v, i| {
            if (i == limit) break;
            if (i > 0) try w.writeByte(',');
            try w.writeByte('[');
            try writeStr(w, k);
            try w.print(",{d}]", .{v});
        }
        try w.writeByte(']');
    }

    /// Totals, `bucket_ms` buckets aligned at `tz_ms` (the viewer's UTC offset), the
    /// sessions active in [from, to) newest first, and the breakdowns.
    pub fn summaryJson(self: *Store, arena: Allocator, w: *std.Io.Writer, from_ms: i64, to_ms: i64, bucket_ms: i64, tz_ms: i64) !void {
        _ = std.c.pthread_mutex_lock(&self.mu);
        defer _ = std.c.pthread_mutex_unlock(&self.mu);
        const bucket = @max(bucket_ms, 60_000);
        const b0 = @divFloor(from_ms + tz_ms, bucket);
        const b1 = @divFloor(@max(from_ms, to_ms - 1) + tz_ms, bucket);
        const n_buckets: usize = @intCast(@min(@max(b1 - b0 + 1, 1), 5000));
        const Bucket = struct { roles: [4]u32 = .{ 0, 0, 0, 0 }, gen: u64 = 0, prompt: u64 = 0, cached: u64 = 0 };
        const buckets = try arena.alloc(Bucket, n_buckets);
        @memset(buckets, .{});

        var totals: Totals = .{};
        var sessions: std.StringArrayHashMapUnmanaged(SessAgg) = .empty;
        var threads: std.AutoHashMapUnmanaged(struct { u32, u64 }, void) = .empty;
        var roles: [4]u64 = .{ 0, 0, 0, 0 };
        var tools: std.StringArrayHashMapUnmanaged(u64) = .empty;
        var clients: std.StringArrayHashMapUnmanaged(u64) = .empty;
        var stops: std.StringArrayHashMapUnmanaged(u64) = .empty;
        var routes: std.StringArrayHashMapUnmanaged(u64) = .empty;

        // Sorted by start: binary-search the window.
        const items = self.records.items;
        var lo: usize = 0;
        var hi: usize = items.len;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (items[mid].ts_ms < from_ms) lo = mid + 1 else hi = mid;
        }
        for (items[lo..]) |*r| {
            if (r.ts_ms >= to_ms) break;
            totals.add(r);
            roles[@intFromEnum(r.role)] += 1;
            for (r.tool_calls) |name| (try tools.getOrPutValue(arena, name, 0)).value_ptr.* += 1;
            (try clients.getOrPutValue(arena, if (r.client.len > 0) r.client else "(none)", 0)).value_ptr.* += 1;
            (try stops.getOrPutValue(arena, if (r.err.len > 0) r.err else if (r.stop.len > 0) r.stop else "(none)", 0)).value_ptr.* += 1;
            (try routes.getOrPutValue(arena, r.route, 0)).value_ptr.* += 1;

            const gp = try sessions.getOrPut(arena, sessionKey(r));
            if (!gp.found_existing) gp.value_ptr.* = .{ .id = gp.key_ptr.* };
            const s = gp.value_ptr;
            const si: u32 = @intCast(gp.index);
            s.requests += 1;
            s.roles[@intFromEnum(r.role)] += 1;
            s.first = @min(s.first, r.ts_ms);
            s.last = @max(s.last, r.ts_ms);
            s.last_end = @max(s.last_end, r.ts_ms + r.dur_ms);
            if (r.title.len > 0 and (!s.title_main or s.title.len == 0) and (r.role == .main or s.title.len == 0)) {
                if (!s.title_main) {
                    s.title = r.title;
                    s.title_main = r.role == .main;
                }
            }
            if (s.client.len == 0) s.client = r.client;
            if (s.device.len == 0) s.device = r.device;
            if (s.peer.len == 0) s.peer = r.peer;
            if (!(try threads.getOrPut(arena, .{ si, r.thread })).found_existing) s.threads += 1;
            s.prompt += r.prompt;
            s.cached += r.cached;
            s.gen += r.gen;
            s.prefill_ms += r.prefill_ms;
            s.decode_ms += r.decode_ms;
            s.busy_ms += r.dur_ms;
            s.calls += @intCast(r.tool_calls.len);
            s.ws += r.web_searches;
            s.images += r.images;
            s.docs += r.docs;
            if (r.err.len > 0) s.errors += 1;

            const bi_raw = @divFloor(r.ts_ms + tz_ms, bucket) - b0;
            if (bi_raw >= 0 and bi_raw < n_buckets) {
                const b = &buckets[@intCast(bi_raw)];
                b.roles[@intFromEnum(r.role)] += 1;
                b.gen += r.gen;
                b.prompt += r.prompt;
                b.cached += r.cached;
            }
        }

        try w.print("{{\"from\":{d},\"to\":{d},\"bucket\":{d},\"tz\":{d},\"first_bucket\":{d},\"held\":{d},\"oldest\":{d},\"totals\":", .{
            from_ms, to_ms, bucket, tz_ms, b0 * bucket - tz_ms, items.len, if (items.len > 0) items[0].ts_ms else 0,
        });
        try totals.write(w);
        try w.print(",\"sessions_n\":{d},\"roles\":{{\"main\":{d},\"subagent\":{d},\"side\":{d},\"other\":{d}}},\"buckets\":[", .{ sessions.count(), roles[0], roles[1], roles[2], roles[3] });
        for (buckets, 0..) |b, i| {
            if (i > 0) try w.writeByte(',');
            // [main, subagent, side, other, gen, prompt, cached]
            try w.print("[{d},{d},{d},{d},{d},{d},{d}]", .{ b.roles[0], b.roles[1], b.roles[2], b.roles[3], b.gen, b.prompt, b.cached });
        }
        try w.writeAll("],");
        try writeCounts(w, "tools", &tools, 25);
        try w.writeByte(',');
        try writeCounts(w, "clients", &clients, 10);
        try w.writeByte(',');
        try writeCounts(w, "stops", &stops, 10);
        try w.writeByte(',');
        try writeCounts(w, "routes", &routes, 10);

        // Sessions, most recently active first.
        const SortCtx = struct {
            vals: []const SessAgg,
            pub fn lessThan(c: @This(), a: usize, b: usize) bool {
                return c.vals[a].last > c.vals[b].last;
            }
        };
        sessions.sort(SortCtx{ .vals = sessions.values() });
        try w.writeAll(",\"sessions\":[");
        for (sessions.values(), 0..) |s, i| {
            if (i == 500) break;
            if (i > 0) try w.writeByte(',');
            try w.writeAll("{\"id\":");
            try writeStr(w, s.id);
            try w.writeAll(",\"title\":");
            try writeStr(w, s.title);
            try w.writeAll(",\"client\":");
            try writeStr(w, s.client);
            try w.writeAll(",\"device\":");
            try writeStr(w, s.device[0..@min(s.device.len, 8)]);
            try w.writeAll(",\"peer\":");
            try writeStr(w, s.peer);
            try w.print(",\"first\":{d},\"last\":{d},\"last_end\":{d},\"requests\":{d},\"main\":{d},\"subagent\":{d},\"side\":{d},\"other\":{d},\"threads\":{d},\"prompt\":{d},\"cached\":{d},\"gen\":{d},\"prefill_ms\":{d},\"decode_ms\":{d},\"busy_ms\":{d},\"calls\":{d},\"ws\":{d},\"images\":{d},\"docs\":{d},\"errors\":{d}}}", .{
                s.first,     s.last,      s.last_end,  s.requests, s.roles[0], s.roles[1], s.roles[2],
                s.roles[3],  s.threads,   s.prompt,    s.cached,   s.gen,      s.prefill_ms, s.decode_ms,
                s.busy_ms,   s.calls,     s.ws,        s.images,   s.docs,     s.errors,
            });
        }
        try w.writeAll("]}");
    }

    /// One session's threads and requests (newest 5000), oldest first.
    pub fn sessionJson(self: *Store, arena: Allocator, w: *std.Io.Writer, id: []const u8) !void {
        _ = std.c.pthread_mutex_lock(&self.mu);
        defer _ = std.c.pthread_mutex_unlock(&self.mu);
        var rows = std.ArrayList(*const Record).empty;
        for (self.records.items) |*r| {
            if (std.mem.eql(u8, sessionKey(r), id)) try rows.append(arena, r);
        }
        const shown = rows.items[rows.items.len -| 5000..];

        const ThreadAgg = struct { id: u64, role: Role, title: []const u8, first: i64, last: i64, requests: u32 = 0, gen: u64 = 0, prompt: u64 = 0, cached: u64 = 0, calls: u32 = 0 };
        var threads: std.AutoArrayHashMapUnmanaged(u64, ThreadAgg) = .empty;
        var totals: Totals = .{};
        var tools: std.StringArrayHashMapUnmanaged(u64) = .empty;
        for (shown) |r| {
            totals.add(r);
            for (r.tool_calls) |name| (try tools.getOrPutValue(arena, name, 0)).value_ptr.* += 1;
            const gp = try threads.getOrPut(arena, r.thread);
            if (!gp.found_existing) gp.value_ptr.* = .{ .id = r.thread, .role = r.role, .title = r.title, .first = r.ts_ms, .last = r.ts_ms };
            const t = gp.value_ptr;
            t.requests += 1;
            t.last = @max(t.last, r.ts_ms + r.dur_ms);
            t.gen += r.gen;
            t.prompt += r.prompt;
            t.cached += r.cached;
            t.calls += @intCast(r.tool_calls.len);
        }

        try w.writeAll("{\"id\":");
        try writeStr(w, id);
        try w.writeAll(",\"client\":");
        try writeStr(w, if (shown.len > 0) shown[shown.len - 1].client else "");
        try w.writeAll(",\"peer\":");
        try writeStr(w, if (shown.len > 0) shown[shown.len - 1].peer else "");
        try w.print(",\"requests_total\":{d},\"totals\":", .{rows.items.len});
        try totals.write(w);
        try w.writeByte(',');
        try writeCounts(w, "tools", &tools, 25);
        try w.writeAll(",\"threads\":[");
        for (threads.values(), 0..) |t, i| {
            if (i > 0) try w.writeByte(',');
            try w.print("{{\"id\":\"{x:0>16}\",\"role\":\"{s}\",\"title\":", .{ t.id, @tagName(t.role) });
            try writeStr(w, t.title);
            try w.print(",\"first\":{d},\"last\":{d},\"requests\":{d},\"gen\":{d},\"prompt\":{d},\"cached\":{d},\"calls\":{d}}}", .{ t.first, t.last, t.requests, t.gen, t.prompt, t.cached, t.calls });
        }
        try w.writeAll("],\"requests\":[");
        for (shown, 0..) |r, i| {
            if (i > 0) try w.writeByte(',');
            const ti = threads.getIndex(r.thread).?;
            try w.print("{{\"ts\":{d},\"dur\":{d},\"t\":{d},\"route\":", .{ r.ts_ms, r.dur_ms, ti });
            try writeStr(w, r.route);
            try w.print(",\"msgs\":{d},\"tools\":{d},\"img\":{d},\"doc\":{d},\"rounds\":{d},\"prompt\":{d},\"cached\":{d},\"gen\":{d},\"prefill_ms\":{d},\"decode_ms\":{d},\"stop\":", .{
                r.msgs, r.tools, r.images, r.docs, r.rounds, r.prompt, r.cached, r.gen, r.prefill_ms, r.decode_ms,
            });
            try writeStr(w, r.stop);
            try w.writeAll(",\"calls\":[");
            for (r.tool_calls, 0..) |name, j| {
                if (j > 0) try w.writeByte(',');
                try writeStr(w, name);
            }
            try w.print("],\"ws\":{d},\"status\":{d},\"err\":", .{ r.web_searches, r.status });
            try writeStr(w, r.err);
            try w.writeByte('}');
        }
        try w.writeAll("]}");
    }
};

// ── Tests ────────────────────────────────────────────────────────────────────

const testing = std.testing;

test "usage: Claude Code user_id forms, JSON and legacy" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cu = claudeUser(a, "{\"device_id\":\"1e066abf8ba8\",\"account_uuid\":\"\",\"session_id\":\"936da46e-7bc5\"}").?;
    try testing.expectEqualStrings("936da46e-7bc5", cu.session);
    try testing.expectEqualStrings("1e066abf8ba8", cu.device);
    const old = claudeUser(a, "user_abc123_account_u-1_session_s-9").?;
    try testing.expectEqualStrings("s-9", old.session);
    try testing.expectEqualStrings("abc123", old.device);
    try testing.expect(claudeUser(a, "alice") == null);
    try testing.expect(claudeUser(a, "{\"device_id\":\"x\"}") == null);
}

test "usage: titleFrom strips system reminders, collapses space, cuts on a UTF-8 boundary" {
    var buf: [TITLE_MAX]u8 = undefined;
    try testing.expectEqualStrings("Fix the build", titleFrom(&buf, "<system-reminder>\nrules\n</system-reminder>\n  Fix   the\nbuild  "));
    try testing.expectEqualStrings("", titleFrom(&buf, "<system-reminder>only</system-reminder>"));
    // 119 ASCII bytes then a 2-byte char: the char does not fit and is dropped whole.
    var long: [125]u8 = @splat('x');
    @memcpy(long[119..], "\u{e9}tail");
    const t = titleFrom(&buf, &long);
    try testing.expectEqual(@as(usize, 119), t.len);
    try testing.expect(std.unicode.utf8ValidateSlice(t));
}

fn parseRoot(a: Allocator, body: []const u8) !std.json.ObjectMap {
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, body, .{});
    return v.object;
}

test "usage: identify tells main, subagent and side calls of one Claude Code session apart" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const uid = "\"{\\\"device_id\\\":\\\"dev1\\\",\\\"account_uuid\\\":\\\"\\\",\\\"session_id\\\":\\\"s1\\\"}\"";
    var tb: [TITLE_MAX]u8 = undefined;
    const main_body = "{\"model\":\"q\",\"stream\":true,\"metadata\":{\"user_id\":" ++ uid ++ "},\"tools\":[{},{}]," ++
        "\"system\":[{\"type\":\"text\",\"text\":\"You are Claude Code.\"},{\"type\":\"text\",\"text\":\"You are an interactive agent that helps users\"}]," ++
        "\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"<system-reminder>ctx</system-reminder>\"},{\"type\":\"text\",\"text\":\"Refactor the parser\"}]}," ++
        "{\"role\":\"assistant\",\"content\":\"ok\"},{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":[{\"type\":\"image\"},{\"type\":\"document\"}]}]}]}";
    const m = identify(a, try parseRoot(a, main_body), &tb, true);
    try testing.expectEqual(Role.main, m.role);
    try testing.expectEqualStrings("s1", m.session);
    try testing.expectEqualStrings("dev1", m.device);
    try testing.expectEqualStrings("Refactor the parser", m.title);
    try testing.expectEqual(@as(u32, 3), m.msgs);
    try testing.expectEqual(@as(u32, 2), m.tools);
    try testing.expectEqual(@as(u32, 1), m.images);
    try testing.expectEqual(@as(u32, 1), m.docs);
    try testing.expect(m.stream);

    // The same thread one turn later: one more message, same first turn, same thread.
    const later = identify(a, try parseRoot(a, main_body[0 .. main_body.len - 2] ++ ",{\"role\":\"assistant\",\"content\":\"x\"}]}"), &tb, true);
    try testing.expectEqual(m.thread, later.thread);

    var tb2: [TITLE_MAX]u8 = undefined;
    const sub_body = "{\"stream\":true,\"metadata\":{\"user_id\":" ++ uid ++ "},\"tools\":[{}]," ++
        "\"system\":\"You are an agent for Claude Code, Anthropic's official CLI\",\"messages\":[{\"role\":\"user\",\"content\":\"List the txt files\"}]}";
    const s = identify(a, try parseRoot(a, sub_body), &tb2, true);
    try testing.expectEqual(Role.subagent, s.role);
    try testing.expect(s.thread != m.thread);
    try testing.expectEqualStrings("List the txt files", s.title);

    // Claude Code 2.1.289 puts system messages AFTER the first user turn and adds more as the
    // conversation grows: the thread must not move with them.
    const turn1 = "{\"stream\":true,\"metadata\":{\"user_id\":" ++ uid ++ "},\"tools\":[{}],\"system\":\"You are an interactive agent\"," ++
        "\"messages\":[{\"role\":\"user\",\"content\":\"go\"},{\"role\":\"system\",\"content\":\"# Environment\"}]}";
    const turn2 = "{\"stream\":true,\"metadata\":{\"user_id\":" ++ uid ++ "},\"tools\":[{}],\"system\":\"You are an interactive agent\"," ++
        "\"messages\":[{\"role\":\"user\",\"content\":\"go\"},{\"role\":\"system\",\"content\":\"# Environment\"},{\"role\":\"assistant\",\"content\":\"x\"}," ++
        "{\"role\":\"user\",\"content\":\"more\"},{\"role\":\"system\",\"content\":\"Available agent types\"}]}";
    try testing.expectEqual(identify(a, try parseRoot(a, turn1), &tb2, false).thread, identify(a, try parseRoot(a, turn2), &tb2, false).thread);

    const side_body = "{\"metadata\":{\"user_id\":" ++ uid ++ "},\"system\":\"You are a security monitor watching an interactive agent\",\"messages\":[{\"role\":\"user\",\"content\":\"classify\"}]}";
    try testing.expectEqual(Role.side, identify(a, try parseRoot(a, side_body), &tb2, false).role);
    // Titles off: nothing kept.
    try testing.expectEqualStrings("", identify(a, try parseRoot(a, side_body), &tb2, false).title);

    // OpenAI chat without a session: role other, session from `user`, system message in the thread.
    const chat_body = "{\"user\":\"bob\",\"messages\":[{\"role\":\"system\",\"content\":\"S\"},{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"},{\"type\":\"image_url\"}]}]}";
    const c = identify(a, try parseRoot(a, chat_body), &tb2, true);
    try testing.expectEqual(Role.other, c.role);
    try testing.expectEqualStrings("bob", c.session);
    try testing.expectEqual(@as(u32, 1), c.images);
    try testing.expectEqualStrings("hi", c.title);
}

test "usage: admitKey groups a Claude Code device, falls back to the peer, exempts helper calls" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const uid1 = "\"{\\\"device_id\\\":\\\"devA\\\",\\\"account_uuid\\\":\\\"\\\",\\\"session_id\\\":\\\"s1\\\"}\"";
    const uid2 = "\"{\\\"device_id\\\":\\\"devA\\\",\\\"account_uuid\\\":\\\"\\\",\\\"session_id\\\":\\\"s2\\\"}\"";
    const agent1 = try parseRoot(a, "{\"stream\":true,\"tools\":[{}],\"metadata\":{\"user_id\":" ++ uid1 ++ "}}");
    const agent2 = try parseRoot(a, "{\"stream\":true,\"tools\":[{}],\"metadata\":{\"user_id\":" ++ uid2 ++ "}}");
    // Two sessions of one device are one user, wherever they connect from.
    try testing.expectEqual(admitKey(agent1, "100.1.1.1"), admitKey(agent2, "100.2.2.2"));
    try testing.expect(admitKey(agent1, "x") != 0);
    // The classifier / title call of that session is exempt.
    try testing.expectEqual(@as(u64, 0), admitKey(try parseRoot(a, "{\"metadata\":{\"user_id\":" ++ uid1 ++ "}}"), "100.1.1.1"));
    // Other clients: one user per address; a no-tools non-stream call still counts.
    const plain = try parseRoot(a, "{\"messages\":[]}");
    try testing.expectEqual(admitKey(plain, "100.1.1.1"), admitKey(plain, "100.1.1.1"));
    try testing.expect(admitKey(plain, "100.1.1.1") != admitKey(plain, "100.2.2.2"));
    try testing.expect(admitKey(plain, "100.1.1.1") != 0);
}

test "usage: a record survives the JSON line round trip" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const calls = [_][]const u8{ "Read", "Bash" };
    const r: Record = .{
        .ts_ms = 1_790_000_000_123, .dur_ms = 4200, .route = "messages", .session = "s1", .device = "dev1",
        .client = "claude-cli/2.1 (external, cli)", .peer = "100.1.2.3", .thread = 0xfeedbeef01, .role = .subagent,
        .title = "Fix \"quotes\"\nand \\ slashes", .model = "qwen3.8", .stream = true, .msgs = 7, .tools = 21, .images = 1,
        .docs = 2, .rounds = 1, .prompt = 45866, .cached = 40960, .gen = 521, .prefill_ms = 5400, .decode_ms = 6800,
        .stop = "tool_use", .tool_calls = &calls, .web_searches = 3, .status = 200, .err = "",
    };
    var buf: [2048]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writeLine(&w, &r);
    const line = w.buffered();
    try testing.expect(line[line.len - 1] == '\n');
    const back = parseLine(arena.allocator(), line[0 .. line.len - 1]).?;
    try testing.expectEqual(r.ts_ms, back.ts_ms);
    try testing.expectEqual(r.thread, back.thread);
    try testing.expectEqual(r.role, back.role);
    try testing.expectEqualStrings(r.title, back.title);
    try testing.expectEqualStrings(r.client, back.client);
    try testing.expectEqual(r.cached, back.cached);
    try testing.expectEqual(@as(usize, 2), back.tool_calls.len);
    try testing.expectEqualStrings("Bash", back.tool_calls[1]);
    try testing.expectEqual(r.web_searches, back.web_searches);
    try testing.expect(parseLine(arena.allocator(), "not json") == null);
}

test "usage: store appends, reloads from disk, and answers the summary and session queries" {
    var dir_buf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrint(&dir_buf, "/tmp/mlx-serve-usage-test-{d}", .{std.c.getpid()});
    const now: i64 = 1_790_000_000_000;
    const calls = [_][]const u8{"Read"};
    {
        const st = try Store.init(testing.allocator, dir, 90, now);
        defer st.deinit();
        st.append(.{ .ts_ms = now - 3000, .dur_ms = 900, .route = "messages", .session = "s1", .client = "cc", .thread = 1, .role = .main, .title = "Main task", .prompt = 100, .cached = 60, .gen = 10, .tool_calls = &calls });
        st.append(.{ .ts_ms = now - 2000, .dur_ms = 500, .route = "messages", .session = "s1", .client = "cc", .thread = 2, .role = .subagent, .title = "Sub task", .prompt = 50, .cached = 0, .gen = 5 });
        // Finishes first but started earliest: the list stays in start order.
        st.append(.{ .ts_ms = now - 9000, .dur_ms = 100, .route = "chat", .session = "", .client = "curl", .thread = 3, .role = .other, .prompt = 10, .gen = 1, .err = "invalid_request_error", .status = 400 });
        st.append(.{ .ts_ms = now - 100 * std.time.ms_per_day, .route = "chat", .client = "old" }); // past the window: written, not kept
        try testing.expectEqual(@as(usize, 3), st.count());
    }
    const st = try Store.init(testing.allocator, dir, 90, now);
    defer {
        st.deinit();
        var p: [256]u8 = undefined;
        var m = monthOf(now);
        for (0..5) |_| {
            const path = std.fmt.bufPrintSentinel(&p, "{s}/usage-{d}-{d:0>2}.jsonl", .{ dir, m / 100, m % 100 }, 0) catch unreachable;
            _ = std.c.unlink(path.ptr);
            m = prevMonth(m);
        }
        const dz = std.fmt.bufPrintSentinel(&p, "{s}", .{dir}, 0) catch unreachable;
        _ = std.c.rmdir(dz.ptr);
    }
    try testing.expectEqual(@as(usize, 3), st.count());
    try testing.expect(st.records.items[0].ts_ms == now - 9000);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    try st.summaryJson(arena.allocator(), &out.writer, now - std.time.ms_per_day, now + 1, std.time.ms_per_hour, 0);
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), out.written(), .{});
    const totals = v.object.get("totals").?.object;
    try testing.expectEqual(@as(i64, 3), totals.get("requests").?.integer);
    try testing.expectEqual(@as(i64, 160), totals.get("prompt").?.integer);
    try testing.expectEqual(@as(i64, 1), totals.get("errors").?.integer);
    const sessions = v.object.get("sessions").?.array.items;
    try testing.expectEqual(@as(usize, 2), sessions.len);
    // Newest first: s1 (last request at now-2000), titled by its main thread.
    try testing.expectEqualStrings("s1", sessions[0].object.get("id").?.string);
    try testing.expectEqualStrings("Main task", sessions[0].object.get("title").?.string);
    try testing.expectEqual(@as(i64, 2), sessions[0].object.get("threads").?.integer);
    try testing.expectEqual(@as(i64, 1), sessions[0].object.get("subagent").?.integer);
    // The anonymous request groups under its client.
    try testing.expectEqualStrings("curl", sessions[1].object.get("id").?.string);
    try testing.expectEqualStrings("Read", v.object.get("tools").?.array.items[0].array.items[0].string);

    out.clearRetainingCapacity();
    try st.sessionJson(arena.allocator(), &out.writer, "s1");
    const sv = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), out.written(), .{});
    try testing.expectEqual(@as(usize, 2), sv.object.get("threads").?.array.items.len);
    const reqs = sv.object.get("requests").?.array.items;
    try testing.expectEqual(@as(usize, 2), reqs.len);
    try testing.expectEqual(@as(i64, 1), reqs[1].object.get("t").?.integer);
}
