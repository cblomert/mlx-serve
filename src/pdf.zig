//! Anthropic `document` content blocks rendered as prompt text.
//!
//! A base64 PDF (`source.type == "base64"`, `media_type == "application/pdf"`
//! — what Claude Code's Read tool sends for a whole-file PDF read) is turned
//! into its text layer by PDFKit (lib/pdftext/pdftext.m; macOS builds only,
//! elsewhere the block renders a note instead), one `--- page N ---` section
//! per page. Plain-text sources (`text`, base64 `text/plain`, `content`
//! arrays) are inlined as they are. Every document renders as
//!
//!     <document title="…" pages="N">
//!     …
//!     </document>
//!
//! Extraction is cached per payload, so a document that stays in a
//! conversation renders byte-identically every turn (the prefix cache keeps
//! matching) and is parsed once, not once per request.

const std = @import("std");
const build_options = @import("build_options");

const Allocator = std.mem.Allocator;

const have_pdfkit = build_options.macos_engines;

const c = if (have_pdfkit) struct {
    extern fn mlxs_pdf_text(data: [*]const u8, len: usize, pages_out: *c_int, text_pages_out: *c_int, len_out: *usize) ?[*]u8;
    extern fn mlxs_pdf_free(p: ?[*]u8) void;
} else struct {};

pub const Extracted = struct {
    /// Page-sectioned text, owned by the allocator passed to the call.
    text: []const u8,
    pages: u32,
    /// Pages with any text; 0 for a scanned (image-only) PDF.
    text_pages: u32,
};

pub const Error = error{ Unsupported, Unreadable, OutOfMemory };

/// Text layer of `pdf`. `error.Unreadable` for bytes PDFKit cannot open (or an
/// encrypted file), `error.Unsupported` on builds without PDFKit.
pub fn extract(allocator: Allocator, pdf: []const u8) Error!Extracted {
    if (comptime !have_pdfkit) return error.Unsupported;
    if (pdf.len < 5 or !std.mem.startsWith(u8, pdf, "%PDF")) return error.Unreadable;
    var pages: c_int = 0;
    var text_pages: c_int = 0;
    var n: usize = 0;
    const p = c.mlxs_pdf_text(pdf.ptr, pdf.len, &pages, &text_pages, &n) orelse return error.Unreadable;
    defer c.mlxs_pdf_free(p);
    return .{
        .text = try allocator.dupe(u8, p[0..n]),
        .pages = @intCast(@max(pages, 0)),
        .text_pages = @intCast(@max(text_pages, 0)),
    };
}

// ── Process-wide cache, keyed by the base64 payload ─────────────────────────

const CACHE_SLOTS = 8;
/// Bytes of extracted text the cache may hold in total.
const CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;

const Slot = struct {
    key: u64,
    len: usize,
    text: []u8,
    pages: u32,
    text_pages: u32,
    stamp: u64,
};

/// pthread mutex: lookups come from connection threads without an Io handle.
var cache_mu: std.c.pthread_mutex_t = .{};
var cache: [CACHE_SLOTS]?Slot = @splat(null);
var cache_clock: u64 = 0;

fn cacheGet(allocator: Allocator, key: u64, len: usize) Error!?Extracted {
    _ = std.c.pthread_mutex_lock(&cache_mu);
    defer _ = std.c.pthread_mutex_unlock(&cache_mu);
    for (&cache) |*slot| if (slot.*) |*s| {
        if (s.key != key or s.len != len) continue;
        cache_clock += 1;
        s.stamp = cache_clock;
        return .{ .text = try allocator.dupe(u8, s.text), .pages = s.pages, .text_pages = s.text_pages };
    };
    return null;
}

fn cachePut(key: u64, len: usize, e: Extracted) void {
    if (e.text.len > CACHE_MAX_BYTES / 2) return;
    const owned = std.heap.c_allocator.dupe(u8, e.text) catch return;
    _ = std.c.pthread_mutex_lock(&cache_mu);
    defer _ = std.c.pthread_mutex_unlock(&cache_mu);
    cache_clock += 1;
    // Evict least-recently-used entries until the new one fits.
    while (true) {
        var total: usize = owned.len;
        var free_slot: ?usize = null;
        var lru: ?usize = null;
        for (cache, 0..) |slot, i| {
            if (slot) |s| {
                total += s.text.len;
                if (lru == null or s.stamp < cache[lru.?].?.stamp) lru = i;
            } else if (free_slot == null) free_slot = i;
        }
        if (free_slot != null and total <= CACHE_MAX_BYTES) {
            cache[free_slot.?] = .{ .key = key, .len = len, .text = owned, .pages = e.pages, .text_pages = e.text_pages, .stamp = cache_clock };
            return;
        }
        const victim = lru orelse {
            std.heap.c_allocator.free(owned);
            return;
        };
        std.heap.c_allocator.free(cache[victim].?.text);
        cache[victim] = null;
    }
}

/// `extract` of a base64 PDF, through the cache.
pub fn extractBase64Cached(allocator: Allocator, b64: []const u8) Error!Extracted {
    const key = std.hash.Wyhash.hash(0x7064_6674, b64);
    if (try cacheGet(allocator, key, b64.len)) |hit| return hit;
    const bytes = decodeBase64(allocator, b64) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.Unreadable,
    };
    defer allocator.free(bytes);
    const e = try extract(allocator, bytes);
    cachePut(key, b64.len, e);
    return e;
}

fn decodeBase64(allocator: Allocator, b64: []const u8) ![]u8 {
    const dec = std.base64.standard.decoderWithIgnore(" \t\r\n");
    const buf = try allocator.alloc(u8, dec.calcSizeUpperBound(b64.len));
    errdefer allocator.free(buf);
    const n = try dec.decode(buf, b64);
    return allocator.realloc(buf, n) catch buf[0..n];
}

// ── Rendering ───────────────────────────────────────────────────────────────

fn strField(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

/// An attribute value with its double quotes swapped for single ones, on one line.
fn writeAttr(w: *std.Io.Writer, name: []const u8, value: []const u8) !void {
    try w.print(" {s}=\"", .{name});
    for (value) |ch| try w.writeByte(switch (ch) {
        '"' => '\'',
        '\r', '\n', '\t' => ' ',
        else => ch,
    });
    try w.writeByte('"');
}

/// A document's content before it is wrapped: text, a page count, and/or a
/// bracketed note for the model (unreadable, scanned, unsupported).
const Body = struct {
    text: []const u8 = "",
    text_owned: bool = false,
    pages: ?u32 = null,
    note: ?[]const u8 = null,
    note_owned: bool = false,

    fn deinit(b: *Body, allocator: Allocator) void {
        if (b.text_owned) allocator.free(b.text);
        if (b.note_owned) allocator.free(b.note.?);
    }
};

fn pdfBody(allocator: Allocator, b64: []const u8) Allocator.Error!Body {
    const e = extractBase64Cached(allocator, b64) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.Unsupported => .{ .note = "[PDF text extraction is not available on this server build.]" },
        error.Unreadable => .{ .note = "[The PDF could not be read: invalid data or password-protected.]" },
    };
    // Page markers alone say nothing; the note does.
    if (e.text_pages == 0) {
        allocator.free(e.text);
        return .{ .pages = e.pages, .note = "[This PDF has no text layer (scanned pages?); its content could not be read as text.]" };
    }
    return .{ .text = e.text, .text_owned = true, .pages = e.pages };
}

fn textB64Body(allocator: Allocator, b64: []const u8) Allocator.Error!Body {
    const bytes = decodeBase64(allocator, b64) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => .{ .note = "[The document's base64 data could not be decoded.]" },
    };
    return .{ .text = bytes, .text_owned = true };
}

fn writeDocument(allocator: Allocator, title: ?[]const u8, context: ?[]const u8, body: Body) ![]u8 {
    var w: std.Io.Writer.Allocating = .init(allocator);
    errdefer w.deinit();
    const out = &w.writer;
    try out.writeAll("<document");
    if (title) |t| if (t.len > 0) try writeAttr(out, "title", t);
    if (body.pages) |p| try out.print(" pages=\"{d}\"", .{p});
    try out.writeAll(">\n");
    if (context) |ctx| if (ctx.len > 0) try out.print("Context: {s}\n", .{ctx});
    if (body.note) |n| try out.print("{s}\n", .{n});
    const trimmed = std.mem.trim(u8, body.text, " \t\r\n");
    if (trimmed.len > 0) {
        try out.writeAll(trimmed);
        try out.writeByte('\n');
    }
    try out.writeAll("</document>");
    return w.toOwnedSlice();
}

/// The prompt text for one Anthropic `document` block (`block` is the block
/// object), allocated with `allocator`. Never fails on bad input: an
/// unreadable or unsupported document renders a bracketed note, so the model
/// can tell the user rather than answer as if it had read it.
pub fn renderDocument(allocator: Allocator, block: std.json.ObjectMap) ![]u8 {
    const src: ?std.json.ObjectMap = if (block.get("source")) |v| (if (v == .object) v.object else null) else null;
    const src_type = if (src) |s| strField(s, "type") orelse "" else "";
    const media = if (src) |s| strField(s, "media_type") orelse "" else "";
    var body: Body = .{};
    defer body.deinit(allocator);
    if (std.mem.eql(u8, src_type, "base64") and std.mem.startsWith(u8, media, "text/")) {
        body = try textB64Body(allocator, strField(src.?, "data") orelse "");
    } else if (std.mem.eql(u8, src_type, "base64")) {
        body = try pdfBody(allocator, strField(src.?, "data") orelse "");
    } else if (std.mem.eql(u8, src_type, "text")) {
        body = .{ .text = strField(src.?, "data") orelse "" };
    } else if (std.mem.eql(u8, src_type, "content")) {
        var joined: std.Io.Writer.Allocating = .init(allocator);
        defer joined.deinit();
        if (src.?.get("content")) |cv| switch (cv) {
            .string => |s| try joined.writer.writeAll(s),
            .array => |arr| for (arr.items) |part| {
                if (part != .object) continue;
                const t = strField(part.object, "text") orelse continue;
                if (joined.written().len > 0) try joined.writer.writeByte('\n');
                try joined.writer.writeAll(t);
            },
            else => {},
        };
        body = .{ .text = try allocator.dupe(u8, joined.written()), .text_owned = true };
    } else if (std.mem.eql(u8, src_type, "url")) {
        body = .{
            .note = try std.fmt.allocPrint(allocator, "[Document not loaded: this server does not fetch document URLs ({s}).]", .{strField(src.?, "url") orelse ""}),
            .note_owned = true,
        };
    } else if (std.mem.eql(u8, src_type, "file")) {
        body = .{ .note = "[Document not loaded: Files API references are not supported by this server.]" };
    } else {
        body = .{ .note = "[Document not loaded: unsupported source.]" };
    }
    return writeDocument(allocator, strField(block, "title"), strField(block, "context"), body);
}

/// An OpenAI file part's `file_data` — a data URL (`data:application/pdf;base64,…`,
/// `data:text/plain;base64,…`) or bare base64, taken as a PDF — rendered like
/// `renderDocument`, titled with the file name. (`file` parts of
/// /v1/chat/completions and `input_file` parts of /v1/responses.)
pub fn renderFileData(allocator: Allocator, file_data: []const u8, filename: ?[]const u8) ![]u8 {
    var media: []const u8 = "application/pdf";
    var b64 = file_data;
    if (std.mem.startsWith(u8, file_data, "data:")) {
        const comma = std.mem.indexOfScalar(u8, file_data, ',') orelse file_data.len;
        const meta = file_data["data:".len..comma];
        media = meta[0 .. std.mem.indexOfScalar(u8, meta, ';') orelse meta.len];
        b64 = if (comma < file_data.len) file_data[comma + 1 ..] else "";
    }
    var body: Body = if (std.mem.startsWith(u8, media, "text/"))
        try textB64Body(allocator, b64)
    else
        try pdfBody(allocator, b64);
    defer body.deinit(allocator);
    return writeDocument(allocator, filename, null, body);
}

/// The document text of an OpenAI-style file part (`{"type":"file","file":{…}}`
/// or `{"type":"input_file",…}`), or null when it carries no inline data
/// (a `file_id` reference: this server has no Files API).
pub fn renderOpenAIFilePart(allocator: Allocator, part: std.json.ObjectMap) !?[]u8 {
    const holder: std.json.ObjectMap = if (part.get("file")) |f| (if (f == .object) f.object else part) else part;
    const data = strField(holder, "file_data") orelse {
        const id = strField(holder, "file_id") orelse return null;
        var body: Body = .{ .note = try std.fmt.allocPrint(allocator, "[File {s} not loaded: this server has no Files API; send file_data inline.]", .{id}), .note_owned = true };
        defer body.deinit(allocator);
        return try writeDocument(allocator, strField(holder, "filename"), null, body);
    };
    return try renderFileData(allocator, data, strField(holder, "filename"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

const testing = std.testing;

fn renderJson(json: []const u8) ![]u8 {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    return renderDocument(testing.allocator, parsed.value.object);
}

test "pdf: text, base64 text/plain and content sources render inline with title and context" {
    const a = try renderJson(
        \\{"type":"document","title":"Notes \"v2\"","context":"from the wiki","source":{"type":"text","media_type":"text/plain","data":"  line one\nline two \n"}}
    );
    defer testing.allocator.free(a);
    try testing.expectEqualStrings("<document title=\"Notes 'v2'\">\nContext: from the wiki\nline one\nline two\n</document>", a);
    const b = try renderJson(
        \\{"type":"document","source":{"type":"base64","media_type":"text/plain","data":"aGVsbG8="}}
    );
    defer testing.allocator.free(b);
    try testing.expectEqualStrings("<document>\nhello\n</document>", b);
    const cc = try renderJson(
        \\{"type":"document","source":{"type":"content","content":[{"type":"text","text":"a"},{"type":"image"},{"type":"text","text":"b"}]}}
    );
    defer testing.allocator.free(cc);
    try testing.expectEqualStrings("<document>\na\nb\n</document>", cc);
}

test "pdf: unreadable and unsupported sources render a note, never fail" {
    const a = try renderJson(
        \\{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"bm90IGEgcGRm"}}
    );
    defer testing.allocator.free(a);
    try testing.expect(std.mem.startsWith(u8, a, "<document>\n["));
    try testing.expect(std.mem.endsWith(u8, a, "]\n</document>"));
    const b = try renderJson(
        \\{"type":"document","source":{"type":"url","url":"https://example.com/x.pdf"}}
    );
    defer testing.allocator.free(b);
    try testing.expect(std.mem.indexOf(u8, b, "https://example.com/x.pdf") != null);
    const cc = try renderJson(
        \\{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"%%%"}}
    );
    defer testing.allocator.free(cc);
    try testing.expect(std.mem.indexOf(u8, cc, "could not be read") != null or std.mem.indexOf(u8, cc, "not available") != null);
}

test "pdf: OpenAI file parts render from data URLs; file_id gets a note" {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\[{"type":"file","file":{"filename":"a.txt","file_data":"data:text/plain;base64,aGk="}},
        \\ {"type":"input_file","filename":"b.pdf","file_data":"data:application/pdf;base64,bm9wZQ=="},
        \\ {"type":"file","file":{"file_id":"file-123"}},
        \\ {"type":"file","file":{}}]
    , .{});
    defer parsed.deinit();
    const items = parsed.value.array.items;
    const a = (try renderOpenAIFilePart(testing.allocator, items[0].object)).?;
    defer testing.allocator.free(a);
    try testing.expectEqualStrings("<document title=\"a.txt\">\nhi\n</document>", a);
    const b = (try renderOpenAIFilePart(testing.allocator, items[1].object)).?;
    defer testing.allocator.free(b);
    try testing.expect(std.mem.startsWith(u8, b, "<document title=\"b.pdf\">\n["));
    const cc = (try renderOpenAIFilePart(testing.allocator, items[2].object)).?;
    defer testing.allocator.free(cc);
    try testing.expect(std.mem.indexOf(u8, cc, "file-123") != null);
    try testing.expect((try renderOpenAIFilePart(testing.allocator, items[3].object)) == null);
}

// A two-page PDF with a text layer (Helvetica, "Hello PDF" / "Page two"), built by hand.
const TINY_PDF =
    "%PDF-1.4\n" ++
    "1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n" ++
    "2 0 obj<</Type/Pages/Kids[3 0 R 5 0 R]/Count 2>>endobj\n" ++
    "3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]/Resources<</Font<</F1 7 0 R>>>>/Contents 4 0 R>>endobj\n" ++
    "4 0 obj<</Length 40>>stream\nBT /F1 18 Tf 20 100 Td (Hello PDF) Tj ET\nendstream endobj\n" ++
    "5 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]/Resources<</Font<</F1 7 0 R>>>>/Contents 6 0 R>>endobj\n" ++
    "6 0 obj<</Length 39>>stream\nBT /F1 18 Tf 20 100 Td (Page two) Tj ET\nendstream endobj\n" ++
    "7 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n" ++
    "trailer<</Root 1 0 R>>\n%%EOF\n";

test "pdf: PDFKit extracts page-sectioned text; the cache returns the same rendering" {
    if (comptime !have_pdfkit) return error.SkipZigTest;
    const e = try extract(testing.allocator, TINY_PDF);
    defer testing.allocator.free(e.text);
    try testing.expectEqual(@as(u32, 2), e.pages);
    try testing.expectEqual(@as(u32, 2), e.text_pages);
    try testing.expectEqualStrings("--- page 1 ---\nHello PDF\n\n--- page 2 ---\nPage two", e.text);

    const enc = std.base64.standard.Encoder;
    var b64: [enc.calcSize(TINY_PDF.len)]u8 = undefined;
    const b64s = enc.encode(&b64, TINY_PDF);
    const json = try std.fmt.allocPrint(testing.allocator, "{{\"type\":\"document\",\"title\":\"t\",\"source\":{{\"type\":\"base64\",\"media_type\":\"application/pdf\",\"data\":\"{s}\"}}}}", .{b64s});
    defer testing.allocator.free(json);
    const first = try renderJson(json);
    defer testing.allocator.free(first);
    try testing.expectEqualStrings("<document title=\"t\" pages=\"2\">\n--- page 1 ---\nHello PDF\n\n--- page 2 ---\nPage two\n</document>", first);
    const second = try renderJson(json);
    defer testing.allocator.free(second);
    try testing.expectEqualStrings(first, second);
}
