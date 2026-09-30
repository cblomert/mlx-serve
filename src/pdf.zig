//! Document content parts rendered as prompt text (+ page images).
//!
//! A base64 PDF (Anthropic `document` blocks — what Claude Code's Read tool
//! sends for a whole-file PDF read — and OpenAI `file` / `input_file` parts) is
//! turned into its text layer by PDFKit (lib/pdftext/pdftext.m; macOS builds
//! only, elsewhere the part renders a note instead), one `--- page N ---`
//! section per page. Pages WITHOUT text (scans) are rendered to RGB by
//! CoreGraphics for the vision encoder (`scannedPageImages`, at most
//! `maxImagePages()` per document) when the model has one; the text names the
//! attached pages. Plain-text sources (`text`, base64 `text/plain`, `content`
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
    extern fn mlxs_pdf_text(data: [*]const u8, len: usize, pages_out: *c_int, text_pages_out: *c_int, len_out: *usize, flags_out: *?[*]u8) ?[*]u8;
    extern fn mlxs_pdf_render_page(data: [*]const u8, len: usize, index: c_int, max_side: c_int, w_out: *c_int, h_out: *c_int) ?[*]u8;
    extern fn mlxs_pdf_free(p: ?*anyopaque) void;
} else struct {};

pub const Extracted = struct {
    /// Page-sectioned text.
    text: []const u8,
    /// One byte per page: 1 where the page has text.
    page_has_text: []const u8,
    pages: u32,
    /// Pages with any text; 0 for a scanned (image-only) PDF.
    text_pages: u32,

    pub fn deinit(e: Extracted, allocator: Allocator) void {
        allocator.free(e.text);
        allocator.free(e.page_has_text);
    }
};

pub const Error = error{ Unsupported, Unreadable, OutOfMemory };

/// Text layer of `pdf`, owned by `allocator` (`Extracted.deinit`).
/// `error.Unreadable` for bytes PDFKit cannot open (or an encrypted file),
/// `error.Unsupported` on builds without PDFKit.
pub fn extract(allocator: Allocator, pdf: []const u8) Error!Extracted {
    if (comptime !have_pdfkit) return error.Unsupported;
    if (pdf.len < 5 or !std.mem.startsWith(u8, pdf, "%PDF")) return error.Unreadable;
    var pages: c_int = 0;
    var text_pages: c_int = 0;
    var n: usize = 0;
    var flags: ?[*]u8 = null;
    const p = c.mlxs_pdf_text(pdf.ptr, pdf.len, &pages, &text_pages, &n, &flags) orelse return error.Unreadable;
    defer c.mlxs_pdf_free(p);
    defer c.mlxs_pdf_free(flags);
    const n_pages: u32 = @intCast(@max(pages, 0));
    const text = try allocator.dupe(u8, p[0..n]);
    errdefer allocator.free(text);
    return .{
        .text = text,
        .page_has_text = try allocator.dupe(u8, if (flags) |f| f[0..n_pages] else &.{}),
        .pages = n_pages,
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
    /// Owned by `std.heap.c_allocator`.
    e: Extracted,
    stamp: u64,
};

/// pthread mutex: lookups come from connection threads without an Io handle.
var cache_mu: std.c.pthread_mutex_t = .{};
var cache: [CACHE_SLOTS]?Slot = @splat(null);
var cache_clock: u64 = 0;

fn dupeExtracted(allocator: Allocator, e: Extracted) Allocator.Error!Extracted {
    const text = try allocator.dupe(u8, e.text);
    errdefer allocator.free(text);
    return .{ .text = text, .page_has_text = try allocator.dupe(u8, e.page_has_text), .pages = e.pages, .text_pages = e.text_pages };
}

fn cacheGet(allocator: Allocator, key: u64, len: usize) Error!?Extracted {
    _ = std.c.pthread_mutex_lock(&cache_mu);
    defer _ = std.c.pthread_mutex_unlock(&cache_mu);
    for (&cache) |*slot| if (slot.*) |*s| {
        if (s.key != key or s.len != len) continue;
        cache_clock += 1;
        s.stamp = cache_clock;
        return try dupeExtracted(allocator, s.e);
    };
    return null;
}

fn cachePut(key: u64, len: usize, e: Extracted) void {
    if (e.text.len > CACHE_MAX_BYTES / 2) return;
    const owned = dupeExtracted(std.heap.c_allocator, e) catch return;
    _ = std.c.pthread_mutex_lock(&cache_mu);
    defer _ = std.c.pthread_mutex_unlock(&cache_mu);
    cache_clock += 1;
    // Evict least-recently-used entries until the new one fits.
    while (true) {
        var total: usize = owned.text.len;
        var free_slot: ?usize = null;
        var lru: ?usize = null;
        for (cache, 0..) |slot, i| {
            if (slot) |s| {
                total += s.e.text.len;
                if (lru == null or s.stamp < cache[lru.?].?.stamp) lru = i;
            } else if (free_slot == null) free_slot = i;
        }
        if (free_slot != null and total <= CACHE_MAX_BYTES) {
            cache[free_slot.?] = .{ .key = key, .len = len, .e = owned, .stamp = cache_clock };
            return;
        }
        const victim = lru orelse {
            owned.deinit(std.heap.c_allocator);
            return;
        };
        cache[victim].?.e.deinit(std.heap.c_allocator);
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

// ── Page images (scanned pages) ─────────────────────────────────────────────

pub const MAX_IMAGE_PAGES_DEFAULT: u32 = 20;
/// Long side of a rendered page; the vision preprocessor resizes it to its own
/// pixel budget (~1 MP for Qwen), so this only has to be at least that sharp.
const RENDER_LONG_SIDE: c_int = 1600;

/// Page images per document (`MLX_SERVE_PDF_IMAGE_PAGES`, default 20; 0 = none).
pub fn maxImagePages() u32 {
    const z = std.c.getenv("MLX_SERVE_PDF_IMAGE_PAGES") orelse return MAX_IMAGE_PAGES_DEFAULT;
    return std.fmt.parseInt(u32, std.mem.span(z), 10) catch MAX_IMAGE_PAGES_DEFAULT;
}

/// 0-based indices of the text-less pages, the first `limit` of them.
fn imagePageIndices(allocator: Allocator, e: Extracted, limit: u32) ![]u32 {
    var out = std.ArrayList(u32).empty;
    errdefer out.deinit(allocator);
    for (e.page_has_text, 0..) |has, i| {
        if (out.items.len >= limit) break;
        if (has == 0) try out.append(allocator, @intCast(i));
    }
    return out.toOwnedSlice(allocator);
}

pub const PageImage = struct {
    /// Top-down RGB8, width * height * 3 bytes, owned by the caller's allocator.
    rgb: []u8,
    width: u32,
    height: u32,
    /// 1-based page number.
    page: u32,
};

/// The text-less pages of a base64 PDF rendered for the vision encoder, at most
/// `maxImagePages()`, in page order. Empty when every page has text or the PDF
/// cannot be read. Free each `rgb`, then the slice.
pub fn scannedPageImages(allocator: Allocator, b64: []const u8) ![]PageImage {
    if (comptime !have_pdfkit) return allocator.alloc(PageImage, 0);
    const e = extractBase64Cached(allocator, b64) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return &.{},
    };
    defer e.deinit(allocator);
    if (e.text_pages >= e.pages) return allocator.alloc(PageImage, 0);
    const idx = try imagePageIndices(allocator, e, maxImagePages());
    defer allocator.free(idx);
    if (idx.len == 0) return allocator.alloc(PageImage, 0);
    const bytes = decodeBase64(allocator, b64) catch return allocator.alloc(PageImage, 0);
    defer allocator.free(bytes);
    var out = std.ArrayList(PageImage).empty;
    errdefer {
        for (out.items) |p| allocator.free(p.rgb);
        out.deinit(allocator);
    }
    for (idx) |i| {
        var w: c_int = 0;
        var h: c_int = 0;
        const px = c.mlxs_pdf_render_page(bytes.ptr, bytes.len, @intCast(i), RENDER_LONG_SIDE, &w, &h) orelse continue;
        defer c.mlxs_pdf_free(px);
        const n: usize = @as(usize, @intCast(w)) * @as(usize, @intCast(h)) * 3;
        try out.append(allocator, .{ .rgb = try allocator.dupe(u8, px[0..n]), .width = @intCast(w), .height = @intCast(h), .page = i + 1 });
    }
    return out.toOwnedSlice(allocator);
}

/// True when a base64 PDF has pages without text (so it contributes page
/// images) — the media-presence test that picks which message's media decodes.
pub fn needsPageImages(b64: []const u8) bool {
    const e = extractBase64Cached(std.heap.c_allocator, b64) catch return false;
    defer e.deinit(std.heap.c_allocator);
    return e.text_pages < e.pages;
}

// ── Rendering ───────────────────────────────────────────────────────────────

fn strField(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

/// The base64 payload of a document block that is a base64 PDF, else null.
pub fn pdfBase64OfBlock(block: std.json.ObjectMap) ?[]const u8 {
    const sv = block.get("source") orelse return null;
    if (sv != .object) return null;
    if (!std.mem.eql(u8, strField(sv.object, "type") orelse "", "base64")) return null;
    if (std.mem.startsWith(u8, strField(sv.object, "media_type") orelse "", "text/")) return null;
    return strField(sv.object, "data");
}

/// `file_data` split into its media type (`application/pdf` for bare base64) and payload.
fn splitFileData(file_data: []const u8) struct { media: []const u8, b64: []const u8 } {
    if (!std.mem.startsWith(u8, file_data, "data:")) return .{ .media = "application/pdf", .b64 = file_data };
    const comma = std.mem.indexOfScalar(u8, file_data, ',') orelse file_data.len;
    const meta = file_data["data:".len..comma];
    return .{
        .media = meta[0 .. std.mem.indexOfScalar(u8, meta, ';') orelse meta.len],
        .b64 = if (comma < file_data.len) file_data[comma + 1 ..] else "",
    };
}

fn fileHolder(part: std.json.ObjectMap) std.json.ObjectMap {
    return if (part.get("file")) |f| (if (f == .object) f.object else part) else part;
}

/// The base64 payload of an OpenAI file part carrying an inline PDF, else null.
pub fn pdfBase64OfFilePart(part: std.json.ObjectMap) ?[]const u8 {
    const data = strField(fileHolder(part), "file_data") orelse return null;
    const fd = splitFileData(data);
    if (std.mem.startsWith(u8, fd.media, "text/")) return null;
    return fd.b64;
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

/// 1-based page numbers as ranges: "1-3, 7, 9-10".
fn writePageList(w: *std.Io.Writer, pages: []const u32) !void {
    var i: usize = 0;
    while (i < pages.len) {
        var j = i;
        while (j + 1 < pages.len and pages[j + 1] == pages[j] + 1) j += 1;
        if (i > 0) try w.writeAll(", ");
        if (j > i) try w.print("{d}-{d}", .{ pages[i], pages[j] }) else try w.print("{d}", .{pages[i]});
        i = j + 1;
    }
}

/// The note for a PDF's text-less pages, or null when every page has text.
/// Worded from the same facts every turn (page flags, the limit, whether this
/// model takes images), never from whether this turn decodes the images.
fn scannedNote(allocator: Allocator, e: Extracted, page_images: bool) !?[]u8 {
    if (e.text_pages >= e.pages) return null;
    var missing = std.ArrayList(u32).empty;
    defer missing.deinit(allocator);
    for (e.page_has_text, 0..) |has, i| if (has == 0) try missing.append(allocator, @intCast(i + 1));
    var w: std.Io.Writer.Allocating = .init(allocator);
    errdefer w.deinit();
    const one = missing.items.len == 1;
    try w.writer.writeAll(if (one) "[Page " else "[Pages ");
    try writePageList(&w.writer, missing.items);
    try w.writer.writeAll(if (one) " has no text layer (scanned)" else " have no text layer (scanned)");
    const limit = maxImagePages();
    if (!page_images or limit == 0) {
        try w.writer.writeAll("; their content could not be read as text.]");
    } else if (missing.items.len <= limit) {
        try w.writer.writeAll(if (one) "; it is attached as an image.]" else "; they are attached as images, in page order.]");
    } else {
        try w.writer.writeAll("; pages ");
        try writePageList(&w.writer, missing.items[0..limit]);
        try w.writer.print(" are attached as images, in page order (limit {d} per document); the rest could not be read.]", .{limit});
    }
    return try w.toOwnedSlice();
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

fn pdfBody(allocator: Allocator, b64: []const u8, page_images: bool) !Body {
    const e = extractBase64Cached(allocator, b64) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.Unsupported => .{ .note = "[PDF text extraction is not available on this server build.]" },
        error.Unreadable => .{ .note = "[The PDF could not be read: invalid data or password-protected.]" },
    };
    defer allocator.free(e.page_has_text);
    const note = scannedNote(allocator, e, page_images) catch |err| {
        allocator.free(e.text);
        return err;
    };
    // A document with no text at all: its page markers alone say nothing.
    if (e.text_pages == 0) {
        allocator.free(e.text);
        return .{ .pages = e.pages, .note = note, .note_owned = note != null };
    }
    return .{ .text = e.text, .text_owned = true, .pages = e.pages, .note = note, .note_owned = note != null };
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
/// can tell the user rather than answer as if it had read it. `page_images`:
/// the model takes images, so scanned pages are attached (`scannedPageImages`)
/// and the note says so.
pub fn renderDocument(allocator: Allocator, block: std.json.ObjectMap, page_images: bool) ![]u8 {
    const src: ?std.json.ObjectMap = if (block.get("source")) |v| (if (v == .object) v.object else null) else null;
    const src_type = if (src) |s| strField(s, "type") orelse "" else "";
    const media = if (src) |s| strField(s, "media_type") orelse "" else "";
    var body: Body = .{};
    defer body.deinit(allocator);
    if (std.mem.eql(u8, src_type, "base64") and std.mem.startsWith(u8, media, "text/")) {
        body = try textB64Body(allocator, strField(src.?, "data") orelse "");
    } else if (std.mem.eql(u8, src_type, "base64")) {
        body = try pdfBody(allocator, strField(src.?, "data") orelse "", page_images);
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
pub fn renderFileData(allocator: Allocator, file_data: []const u8, filename: ?[]const u8, page_images: bool) ![]u8 {
    const fd = splitFileData(file_data);
    var body: Body = if (std.mem.startsWith(u8, fd.media, "text/"))
        try textB64Body(allocator, fd.b64)
    else
        try pdfBody(allocator, fd.b64, page_images);
    defer body.deinit(allocator);
    return writeDocument(allocator, filename, null, body);
}

/// The document text of an OpenAI-style file part (`{"type":"file","file":{…}}`
/// or `{"type":"input_file",…}`), or null when it carries nothing (a
/// `file_id` reference renders a note: this server has no Files API).
pub fn renderOpenAIFilePart(allocator: Allocator, part: std.json.ObjectMap, page_images: bool) !?[]u8 {
    const holder = fileHolder(part);
    const data = strField(holder, "file_data") orelse {
        const id = strField(holder, "file_id") orelse return null;
        var body: Body = .{ .note = try std.fmt.allocPrint(allocator, "[File {s} not loaded: this server has no Files API; send file_data inline.]", .{id}), .note_owned = true };
        defer body.deinit(allocator);
        return try writeDocument(allocator, strField(holder, "filename"), null, body);
    };
    return try renderFileData(allocator, data, strField(holder, "filename"), page_images);
}


// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

const testing = std.testing;

fn renderJson(json: []const u8) ![]u8 {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    return renderDocument(testing.allocator, parsed.value.object, true);
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
    const a = (try renderOpenAIFilePart(testing.allocator, items[0].object, true)).?;
    defer testing.allocator.free(a);
    try testing.expectEqualStrings("<document title=\"a.txt\">\nhi\n</document>", a);
    const b = (try renderOpenAIFilePart(testing.allocator, items[1].object, true)).?;
    defer testing.allocator.free(b);
    try testing.expect(std.mem.startsWith(u8, b, "<document title=\"b.pdf\">\n["));
    const cc = (try renderOpenAIFilePart(testing.allocator, items[2].object, true)).?;
    defer testing.allocator.free(cc);
    try testing.expect(std.mem.indexOf(u8, cc, "file-123") != null);
    try testing.expect((try renderOpenAIFilePart(testing.allocator, items[3].object, true)) == null);
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
    defer e.deinit(testing.allocator);
    try testing.expectEqualSlices(u8, &.{ 1, 1 }, e.page_has_text);
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

test "pdf: writePageList joins runs into ranges" {
    var w: std.Io.Writer.Allocating = .init(testing.allocator);
    defer w.deinit();
    try writePageList(&w.writer, &.{ 1, 2, 3, 7, 9, 10 });
    try testing.expectEqualStrings("1-3, 7, 9-10", w.written());
}

// Page 1 has text, page 2 only a filled rectangle: a "scanned" page.
const MIXED_PDF =
    "%PDF-1.4\n" ++
    "1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n" ++
    "2 0 obj<</Type/Pages/Kids[3 0 R 5 0 R]/Count 2>>endobj\n" ++
    "3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]/Resources<</Font<</F1 7 0 R>>>>/Contents 4 0 R>>endobj\n" ++
    "4 0 obj<</Length 40>>stream\nBT /F1 18 Tf 20 100 Td (Hello PDF) Tj ET\nendstream endobj\n" ++
    "5 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]/Contents 6 0 R>>endobj\n" ++
    "6 0 obj<</Length 25>>stream\n0 0 0 rg 0 0 150 200 re f\nendstream endobj\n" ++
    "7 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n" ++
    "trailer<</Root 1 0 R>>\n%%EOF\n";

test "pdf: a text-less page is named in the note and rendered as a page image" {
    if (comptime !have_pdfkit) return error.SkipZigTest;
    const e = try extract(testing.allocator, MIXED_PDF);
    defer e.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 2), e.pages);
    try testing.expectEqualSlices(u8, &.{ 1, 0 }, e.page_has_text);

    const enc = std.base64.standard.Encoder;
    var b64: [enc.calcSize(MIXED_PDF.len)]u8 = undefined;
    const b64s = enc.encode(&b64, MIXED_PDF);
    const json = try std.fmt.allocPrint(testing.allocator, "{{\"type\":\"document\",\"source\":{{\"type\":\"base64\",\"media_type\":\"application/pdf\",\"data\":\"{s}\"}}}}", .{b64s});
    defer testing.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const with_images = try renderDocument(testing.allocator, parsed.value.object, true);
    defer testing.allocator.free(with_images);
    try testing.expectEqualStrings("<document pages=\"2\">\n[Page 2 has no text layer (scanned); it is attached as an image.]\n--- page 1 ---\nHello PDF\n\n--- page 2 ---\n</document>", with_images);
    const text_only = try renderDocument(testing.allocator, parsed.value.object, false);
    defer testing.allocator.free(text_only);
    try testing.expect(std.mem.indexOf(u8, text_only, "could not be read as text") != null);
    try testing.expectEqualStrings(b64s, pdfBase64OfBlock(parsed.value.object).?);
    try testing.expect(needsPageImages(b64s));

    const imgs = try scannedPageImages(testing.allocator, b64s);
    defer {
        for (imgs) |p| testing.allocator.free(p.rgb);
        testing.allocator.free(imgs);
    }
    try testing.expectEqual(@as(usize, 1), imgs.len);
    try testing.expectEqual(@as(u32, 2), imgs[0].page);
    try testing.expectEqual(@as(u32, 1600), imgs[0].width);
    try testing.expectEqual(@as(u32, 1067), imgs[0].height);
    // Left half black (the rectangle), right half white paper; top row first.
    const row = imgs[0].width * 3;
    try testing.expect(imgs[0].rgb[10 * row + 100 * 3] < 16);
    try testing.expect(imgs[0].rgb[10 * row + 1500 * 3] > 240);
}
