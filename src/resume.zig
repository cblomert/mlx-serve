//! Resume on retry: the tokens a generation had produced when its client hung up, keyed by
//! (user, prompt). An identical request that arrives within the TTL continues from them
//! instead of regenerating from token 0.
//!
//! Why: Claude Code abandons a request at API_TIMEOUT_MS (600 s by default) and re-sends the
//! same body (streamed, then non-streamed after ~5 min), and every copy started over: a
//! 15-30k-token subagent turn at ~30 tok/s never finished (live 2026-10-15 and 10-20: up to
//! 1.1M generated tokens a day thrown away). The server-side half: on a disconnect the
//! handler saves what it produced; the retry's slot prefills prompt + saved tokens (the
//! cancelled slot committed exactly that to the hot cache, so it is a cache hit back to the
//! last checkpoint), the saved tokens replay to the client through the normal per-token
//! path, and decoding continues. Each abandoned copy now adds to the answer instead of
//! repeating it.
//!
//! The key is the token-level prompt, not the request body: `stream`, `max_tokens` and
//! headers differ between Claude Code's attempts, the rendered prompt does not. The user
//! (`admitKey`) is in the key so one user never continues another's text.

const std = @import("std");

/// How long saved tokens wait for their retry.
pub const TTL_MS: i64 = 30 * std.time.ms_per_min;
/// Below this a retry regenerates in seconds; not worth an entry.
pub const MIN_TOKENS: usize = 64;
/// Saved generations kept at once (token ids only: 4 bytes per token).
pub const CAPACITY: usize = 32;
/// A retry can race the abandoned copy, which saves only once it notices the hang-up (next
/// token, or next keepalive while prefilling). Wait this long for it before starting over.
pub const IN_FLIGHT_WAIT_MS: u64 = 5000;

pub fn key(user: u64, prompt_ids: []const u32, has_tools: bool, thinking: bool) u64 {
    var h = std.hash.XxHash64.init(0x7265_7375_6d65_0001);
    h.update(std.mem.asBytes(&user));
    h.update(std.mem.sliceAsBytes(prompt_ids));
    const flags: u8 = @as(u8, @intFromBool(has_tools)) | (@as(u8, @intFromBool(thinking)) << 1);
    h.update(&[_]u8{flags});
    return h.final();
}

pub const Store = struct {
    allocator: std.mem.Allocator,
    mu: std.Io.Mutex = .init,
    entries: [CAPACITY]Entry = @splat(.{}),
    /// Requests currently generating under each key (a retry waits for the copy it races).
    live: [CAPACITY]Live = @splat(.{}),

    const Entry = struct { key: u64 = 0, ids: []u32 = &.{}, at_ms: i64 = 0 };
    const Live = struct { key: u64 = 0, n: u32 = 0 };

    pub fn deinit(self: *Store) void {
        for (&self.entries) |*e| self.drop(e);
    }

    fn drop(self: *Store, e: *Entry) void {
        if (e.ids.len > 0) self.allocator.free(e.ids);
        e.* = .{};
    }

    /// Save `ids` under `k`, replacing an older save of the same key, else the oldest entry.
    /// Returns whether it saved (too short, or out of memory, does not).
    pub fn put(self: *Store, io: std.Io, k: u64, ids: []const u32, now_ms: i64) bool {
        if (ids.len < MIN_TOKENS) return false;
        const copy = self.allocator.dupe(u32, ids) catch return false;
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        var slot: *Entry = &self.entries[0];
        for (&self.entries) |*e| {
            if (e.ids.len > 0 and e.key == k) {
                slot = e;
                break;
            }
            if (e.ids.len == 0) {
                if (slot.ids.len > 0) slot = e;
            } else if (slot.ids.len > 0 and e.at_ms < slot.at_ms) {
                slot = e;
            }
        }
        self.drop(slot);
        slot.* = .{ .key = k, .ids = copy, .at_ms = now_ms };
        return true;
    }

    /// Remove and return the tokens saved under `k` (caller frees with `self.allocator`), or
    /// null when there are none or they are older than the TTL.
    pub fn take(self: *Store, io: std.Io, k: u64, now_ms: i64) ?[]u32 {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        for (&self.entries) |*e| {
            if (e.ids.len == 0 or e.key != k) continue;
            if (now_ms - e.at_ms > TTL_MS) {
                self.drop(e);
                return null;
            }
            const ids = e.ids;
            e.* = .{};
            return ids;
        }
        return null;
    }

    /// A request with key `k` started generating. Silently untracked when every slot is taken
    /// (the retry then simply does not wait).
    pub fn enter(self: *Store, io: std.Io, k: u64) void {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        for (&self.live) |*l| {
            if (l.n > 0 and l.key == k) {
                l.n += 1;
                return;
            }
        }
        for (&self.live) |*l| {
            if (l.n == 0) {
                l.* = .{ .key = k, .n = 1 };
                return;
            }
        }
    }

    pub fn leave(self: *Store, io: std.Io, k: u64) void {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        for (&self.live) |*l| {
            if (l.n > 0 and l.key == k) {
                l.n -= 1;
                return;
            }
        }
    }

    pub fn busy(self: *Store, io: std.Io, k: u64) bool {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        for (&self.live) |*l| {
            if (l.n > 0 and l.key == k) return true;
        }
        return false;
    }
};

/// The decision a retry makes: continue from the saved tokens only when they leave room to
/// generate (a prefix at max_tokens would just replay a truncated answer).
pub fn usable(saved_len: usize, max_tokens: u32) bool {
    return saved_len + 16 < max_tokens;
}

const testing = std.testing;

fn seq(comptime n: usize, base: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*t, i| t.* = base + @as(u32, @intCast(i));
    return out;
}

test "resume key: same prompt + user + flags match; any difference does not" {
    const a = seq(100, 7);
    var b = a;
    try testing.expectEqual(key(1, &a, true, true), key(1, &b, true, true));
    b[99] += 1;
    try testing.expect(key(1, &a, true, true) != key(1, &b, true, true));
    try testing.expect(key(1, &a, true, true) != key(2, &a, true, true));
    try testing.expect(key(1, &a, true, true) != key(1, &a, false, true));
    try testing.expect(key(1, &a, true, true) != key(1, &a, true, false));
    try testing.expect(key(1, a[0..99], true, true) != key(1, &a, true, true));
}

test "resume store: put then take returns the tokens once; short saves are ignored" {
    const io = testing.io;
    var s: Store = .{ .allocator = testing.allocator };
    defer s.deinit();
    const gen = seq(200, 1000);
    try testing.expect(s.put(io, 42, &gen, 0));
    try testing.expect(!s.put(io, 43, gen[0 .. MIN_TOKENS - 1], 0));
    const got = s.take(io, 42, 1000).?;
    defer testing.allocator.free(got);
    try testing.expectEqualSlices(u32, &gen, got);
    try testing.expect(s.take(io, 42, 1000) == null);
    try testing.expect(s.take(io, 43, 1000) == null);
}

test "resume store: a later save of the same key replaces the earlier one" {
    const io = testing.io;
    var s: Store = .{ .allocator = testing.allocator };
    defer s.deinit();
    const first = seq(100, 1);
    const longer = seq(300, 1);
    _ = s.put(io, 9, &first, 0);
    _ = s.put(io, 9, &longer, 10);
    const got = s.take(io, 9, 20).?;
    defer testing.allocator.free(got);
    try testing.expectEqual(@as(usize, 300), got.len);
    try testing.expect(s.take(io, 9, 20) == null);
}

test "resume store: saves expire after the TTL" {
    const io = testing.io;
    var s: Store = .{ .allocator = testing.allocator };
    defer s.deinit();
    const gen = seq(100, 1);
    _ = s.put(io, 5, &gen, 1_000);
    try testing.expect(s.take(io, 5, 1_000 + TTL_MS + 1) == null);
    _ = s.put(io, 5, &gen, 1_000);
    const got = s.take(io, 5, 1_000 + TTL_MS).?;
    testing.allocator.free(got);
}

test "resume store: when full, the oldest save makes room" {
    const io = testing.io;
    var s: Store = .{ .allocator = testing.allocator };
    defer s.deinit();
    const gen = seq(100, 1);
    for (0..CAPACITY) |i| _ = s.put(io, 100 + i, &gen, @intCast(1000 + i));
    _ = s.put(io, 999, &gen, 5000);
    try testing.expect(s.take(io, 100, 6000) == null); // the oldest went
    for (1..CAPACITY) |i| {
        const got = s.take(io, 100 + i, 6000).?;
        testing.allocator.free(got);
    }
    const got = s.take(io, 999, 6000).?;
    testing.allocator.free(got);
}

test "resume store: live counting per key" {
    const io = testing.io;
    var s: Store = .{ .allocator = testing.allocator };
    defer s.deinit();
    try testing.expect(!s.busy(io, 1));
    s.enter(io, 1);
    s.enter(io, 1);
    s.enter(io, 2);
    try testing.expect(s.busy(io, 1));
    s.leave(io, 1);
    try testing.expect(s.busy(io, 1));
    s.leave(io, 1);
    try testing.expect(!s.busy(io, 1));
    try testing.expect(s.busy(io, 2));
    s.leave(io, 2);
    s.leave(io, 2); // unmatched leave is harmless
    try testing.expect(!s.busy(io, 2));
}

test "usable: a saved prefix must leave room to generate" {
    try testing.expect(usable(1000, 32000));
    try testing.expect(!usable(31990, 32000));
    try testing.expect(!usable(32000, 32000));
}
