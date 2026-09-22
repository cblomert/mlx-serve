//! MiMo-V2.6 vision tower (MiMoVisionTransformer), dispatched from
//! `vision.VisionEncoder` when `config.mimo_vision` is set.
//!
//! Spec: the checkpoint's modeling_mimo_v2.py, with two corrections the
//! checkpoint itself settles:
//!   * Sinks are applied ONLY in windowed blocks. The weights ship
//!     `attn.sinks` for the 24 windowed blocks and none for the full blocks
//!     (0, 9, 18, 27) — vLLM's convention, and what an untouched zero-init
//!     parameter in HF's code amounts to.
//!   * A sink is an ADDITIVE BIAS on key 0's logit, not MLX's extra-logit
//!     `sinks` argument. Passing it as an SDPA sink would be silently wrong.
//!
//! Validated first as an MLX prototype (scratchpad/mimo_vision_proto.py),
//! which captions a synthetic image exactly: shapes, colours, positions and
//! text. This file is that prototype in the C API.
//!
//! Layout per block: x += proj(attn(rms_norm1(x))); x += swiglu(rms_norm2(x)).
//! Attention is GQA 32/8, head 64, 2-D rotary, and per block either FULL
//! (bidirectional, no mask) or WINDOWED (|i - j| <= 64 in the current token
//! order). Windowed blocks of type 1 run on a COLUMN-major reordering of the
//! merge units (and column-ordered rope); type 0 and full blocks on row order.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const Weights = model_mod.Weights;

/// Per-block window kind from the checkpoint's `vit_window_attn_types`:
/// -1 full, 0 row-ordered window, 1 column-ordered window.
const WINDOW_TYPES = [_]i8{ -1, 0, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 1, 1, 1, 1, -1 };
const WINDOW: c_int = 64; // visual_token_window_size
const EPS: f32 = 1e-6;
const ROPE_THETA: f64 = 10000.0;

const Block = struct {
    norm1: mlx.mlx_array,
    norm2: mlx.mlx_array,
    qkv_w: mlx.mlx_array,
    qkv_b: mlx.mlx_array,
    proj_w: mlx.mlx_array,
    proj_b: mlx.mlx_array,
    gate_w: mlx.mlx_array,
    gate_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_b: mlx.mlx_array,
    /// [heads] bias on key 0's logit; null handle on full-attention blocks.
    sinks: mlx.mlx_array,
    kind: i8,
};

pub const MimoVision = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,
    weights: Weights,

    hidden: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    merge: u32,
    out_hidden: u32,

    patch_w: mlx.mlx_array, // [hidden, C*tps*ps*ps]
    blocks: []Block,
    merger_ln: mlx.mlx_array, // [hidden] LayerNorm weight, no bias
    merger_fc1: mlx.mlx_array, // [hidden*merge², hidden*merge²], no bias
    merger_fc2: mlx.mlx_array, // [out_hidden, hidden*merge²], no bias

    pub fn initFromDir(allocator: std.mem.Allocator, model_dir: []const u8) !MimoVision {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/omnimodal/vision_encoder.safetensors", .{model_dir});
        var weights = model_mod.loadWeightsSingleFile(allocator, path) catch |err| {
            log.warn("[vision] mimo: cannot load {s} ({s})\n", .{ path, @errorName(err) });
            return error.MissingVisionWeights;
        };
        errdefer weights.deinit();

        const s = mlx.mlx_default_gpu_stream_new();
        var kb: [128]u8 = undefined;
        const get = struct {
            fn f(w: *const Weights, key: []const u8) !mlx.mlx_array {
                return w.get(key) orelse {
                    log.warn("[vision] mimo: missing {s}\n", .{key});
                    return error.MissingVisionWeights;
                };
            }
        }.f;

        // Conv3d [out, C, kT, ps, ps] as a Linear over the processor's
        // [C, tps, py, px] patch features — channels-first, flattened as-is.
        const conv = try get(&weights, "visual.patch_embed.proj.weight");
        const cs = mlx.getShape(conv);
        if (cs.len != 5) return error.MissingVisionWeights;
        var patch_w = mlx.mlx_array_new();
        const flat = [_]c_int{ cs[0], cs[1] * cs[2] * cs[3] * cs[4] };
        try mlx.check(mlx.mlx_reshape(&patch_w, conv, &flat, 2, s));
        const hidden: u32 = @intCast(cs[0]);

        const blocks = try allocator.alloc(Block, WINDOW_TYPES.len);
        errdefer allocator.free(blocks);
        for (blocks, 0..) |*b, i| {
            const k = struct {
                fn f(buf: []u8, li: usize, rest: []const u8) []const u8 {
                    return std.fmt.bufPrint(buf, "visual.blocks.{d}.{s}", .{ li, rest }) catch unreachable;
                }
            }.f;
            b.* = .{
                .norm1 = try get(&weights, k(&kb, i, "norm1.weight")),
                .norm2 = try get(&weights, k(&kb, i, "norm2.weight")),
                .qkv_w = try get(&weights, k(&kb, i, "attn.qkv.weight")),
                .qkv_b = try get(&weights, k(&kb, i, "attn.qkv.bias")),
                .proj_w = try get(&weights, k(&kb, i, "attn.proj.weight")),
                .proj_b = try get(&weights, k(&kb, i, "attn.proj.bias")),
                .gate_w = try get(&weights, k(&kb, i, "mlp.gate_proj.weight")),
                .gate_b = try get(&weights, k(&kb, i, "mlp.gate_proj.bias")),
                .up_w = try get(&weights, k(&kb, i, "mlp.up_proj.weight")),
                .up_b = try get(&weights, k(&kb, i, "mlp.up_proj.bias")),
                .down_w = try get(&weights, k(&kb, i, "mlp.down_proj.weight")),
                .down_b = try get(&weights, k(&kb, i, "mlp.down_proj.bias")),
                .sinks = weights.get(k(&kb, i, "attn.sinks")) orelse .{ .ctx = null },
                .kind = WINDOW_TYPES[i],
            };
            // The checkpoint's own structure: sinks exist on windowed blocks only.
            if ((b.kind == -1) != (b.sinks.ctx == null)) {
                log.warn("[vision] mimo: block {d} kind {d} but sinks {s}\n", .{ i, b.kind, if (b.sinks.ctx == null) "absent" else "present" });
            }
        }

        // Head geometry from the fused qkv rows: (heads + 2·kv) · head_dim.
        // heads · head_dim is the proj input width.
        const proj_in: u32 = @intCast(mlx.getShape(blocks[0].proj_w)[1]);
        const qkv_rows: u32 = @intCast(mlx.getShape(blocks[0].qkv_w)[0]);
        const heads: u32 = @intCast(mlx.getShape(blocks[1].sinks)[0]);
        const head_dim: u32 = proj_in / heads;
        const kv_heads: u32 = (qkv_rows / head_dim - heads) / 2;

        const fc2 = try get(&weights, "visual.merger.mlp.2.weight");
        const out_hidden: u32 = @intCast(mlx.getShape(fc2)[0]);
        const fc1 = try get(&weights, "visual.merger.mlp.0.weight");
        const merge2: u32 = @as(u32, @intCast(mlx.getShape(fc1)[1])) / hidden;
        const merge: u32 = std.math.sqrt(merge2);

        log.info("[vision] mimo tower: depth={d} hidden={d} heads={d}/{d} head_dim={d} merge={d} out={d}\n", .{
            blocks.len, hidden, heads, kv_heads, head_dim, merge, out_hidden,
        });
        return .{
            .s = s,
            .allocator = allocator,
            .weights = weights,
            .hidden = hidden,
            .heads = heads,
            .kv_heads = kv_heads,
            .head_dim = head_dim,
            .merge = merge,
            .out_hidden = out_hidden,
            .patch_w = patch_w,
            .blocks = blocks,
            .merger_ln = try get(&weights, "visual.merger.ln_q.weight"),
            .merger_fc1 = fc1,
            .merger_fc2 = fc2,
        };
    }

    pub fn deinit(self: *MimoVision) void {
        _ = mlx.mlx_array_free(self.patch_w);
        self.allocator.free(self.blocks);
        self.weights.deinit();
    }

    // ── small owned-array helpers ────────────────────────────────────────

    fn linear(self: *MimoVision, x: mlx.mlx_array, w: mlx.mlx_array, b: ?mlx.mlx_array) !mlx.mlx_array {
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, w, self.s));
        var y = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&y, x, wt, self.s));
        if (b) |bias| {
            var yb = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&yb, y, bias, self.s));
            _ = mlx.mlx_array_free(y);
            return yb;
        }
        return y;
    }

    fn scalar(self: *MimoVision, v: f32, dt: mlx.mlx_dtype) !mlx.mlx_array {
        const f = mlx.mlx_array_new_float(v);
        defer _ = mlx.mlx_array_free(f);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&out, f, dt, self.s));
        return out;
    }

    fn sliceAxis(self: *MimoVision, x: mlx.mlx_array, axis: usize, lo: c_int, hi: c_int) !mlx.mlx_array {
        const sh = mlx.getShape(x);
        var start: [4]c_int = .{ 0, 0, 0, 0 };
        var stop: [4]c_int = .{ 0, 0, 0, 0 };
        const strides: [4]c_int = .{ 1, 1, 1, 1 };
        for (sh, 0..) |d, i| stop[i] = d;
        start[axis] = lo;
        stop[axis] = hi;
        var out = mlx.mlx_array_new();
        const nd: usize = sh.len;
        try mlx.check(mlx.mlx_slice(&out, x, &start, nd, &stop, nd, &strides, nd, self.s));
        return out;
    }

    /// [N, heads*D] columns → [1, heads, N, D].
    fn toHeads(self: *MimoVision, x: mlx.mlx_array, n: c_int, heads: u32) !mlx.mlx_array {
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        const shp = [_]c_int{ 1, n, @intCast(heads), @intCast(self.head_dim) };
        try mlx.check(mlx.mlx_reshape(&r, x, &shp, 4, self.s));
        var t = mlx.mlx_array_new();
        const perm = [_]c_int{ 0, 2, 1, 3 };
        try mlx.check(mlx.mlx_transpose_axes(&t, r, &perm, 4, self.s));
        return t;
    }

    /// Non-interleaved rotary over the full head (rotate_half), in f32, back
    /// to the input dtype — the reference's `_apply_rotary_pos_emb_vision`.
    fn rope(self: *MimoVision, x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array) !mlx.mlx_array {
        const dt = mlx.mlx_array_dtype(x);
        var xf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xf);
        try mlx.check(mlx.mlx_astype(&xf, x, .float32, self.s));
        const half: c_int = @intCast(self.head_dim / 2);
        const x1 = try self.sliceAxis(xf, 3, 0, half);
        defer _ = mlx.mlx_array_free(x1);
        const x2 = try self.sliceAxis(xf, 3, half, half * 2);
        defer _ = mlx.mlx_array_free(x2);
        var nx2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(nx2);
        try mlx.check(mlx.mlx_negative(&nx2, x2, self.s));
        var rot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rot);
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, nx2);
            _ = mlx.mlx_vector_array_append_value(vec, x1);
            try mlx.check(mlx.mlx_concatenate_axis(&rot, vec, 3, self.s));
        }
        var a = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a);
        try mlx.check(mlx.mlx_multiply(&a, xf, cos, self.s));
        var b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(b);
        try mlx.check(mlx.mlx_multiply(&b, rot, sin, self.s));
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, a, b, self.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&out, sum, dt, self.s));
        return out;
    }

    const Masks = struct {
        band: mlx.mlx_array, // [1,1,N,N]
        band_top: mlx.mlx_array, // [1,1,T,N]
        band_rest: mlx.mlx_array, // [1,1,N-T,N] (null when N <= T)
        key0: mlx.mlx_array, // [1,1,1,N] one-hot at key 0
        top: c_int,

        fn deinit(m: *Masks) void {
            _ = mlx.mlx_array_free(m.band);
            _ = mlx.mlx_array_free(m.band_top);
            if (m.band_rest.ctx != null) _ = mlx.mlx_array_free(m.band_rest);
            _ = mlx.mlx_array_free(m.key0);
        }
    };

    /// Additive window band `|i - j| > WINDOW → -inf`, shared by every
    /// windowed block of one forward. The sink bias only reaches queries whose
    /// window still contains key 0 (rows 0..WINDOW); every later row masks key
    /// 0 to -inf regardless. So those rows get their own per-head mask and
    /// the rest share ONE [N, N] band — exact, and it keeps the mask at N²
    /// instead of heads·N².
    fn buildMasks(self: *MimoVision, n: c_int, dt: mlx.mlx_dtype) !Masks {
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_arange(&r, 0, @floatFromInt(n), 1, .float32, self.s));
        var rc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rc);
        const col_shape = [_]c_int{ n, 1 };
        try mlx.check(mlx.mlx_reshape(&rc, r, &col_shape, 2, self.s));
        var rr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rr);
        const row_shape = [_]c_int{ 1, n };
        try mlx.check(mlx.mlx_reshape(&rr, r, &row_shape, 2, self.s));
        var d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d);
        try mlx.check(mlx.mlx_subtract(&d, rc, rr, self.s));
        var ad = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ad);
        try mlx.check(mlx.mlx_abs(&ad, d, self.s));
        const w = mlx.mlx_array_new_float(@floatFromInt(WINDOW));
        defer _ = mlx.mlx_array_free(w);
        var cond = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cond);
        try mlx.check(mlx.mlx_greater(&cond, ad, w, self.s));
        const ninf = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(ninf);
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);
        var band2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(band2);
        try mlx.check(mlx.mlx_where(&band2, cond, ninf, zero, self.s));
        var band2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(band2d);
        try mlx.check(mlx.mlx_astype(&band2d, band2, dt, self.s));
        var band = mlx.mlx_array_new();
        const b4 = [_]c_int{ 1, 1, n, n };
        try mlx.check(mlx.mlx_reshape(&band, band2d, &b4, 4, self.s));

        const top: c_int = @min(n, WINDOW + 1);
        const band_top = try self.sliceAxis(band, 2, 0, top);
        const band_rest: mlx.mlx_array = if (n > top) try self.sliceAxis(band, 2, top, n) else .{ .ctx = null };

        // key0 one-hot [1,1,1,N]
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_equal(&eq, rr, zero, self.s));
        var eqd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eqd);
        try mlx.check(mlx.mlx_astype(&eqd, eq, dt, self.s));
        var key0 = mlx.mlx_array_new();
        const k4 = [_]c_int{ 1, 1, 1, n };
        try mlx.check(mlx.mlx_reshape(&key0, eqd, &k4, 4, self.s));
        return .{ .band = band, .band_top = band_top, .band_rest = band_rest, .key0 = key0, .top = top };
    }

    fn attention(self: *MimoVision, h: mlx.mlx_array, blk: *const Block, cos: mlx.mlx_array, sin: mlx.mlx_array, masks: *const Masks, n: c_int) !mlx.mlx_array {
        const qkv = try self.linear(h, blk.qkv_w, blk.qkv_b);
        defer _ = mlx.mlx_array_free(qkv);
        const qd: c_int = @intCast(self.heads * self.head_dim);
        const kd: c_int = @intCast(self.kv_heads * self.head_dim);
        const q_c = try self.sliceAxis(qkv, 1, 0, qd);
        defer _ = mlx.mlx_array_free(q_c);
        const k_c = try self.sliceAxis(qkv, 1, qd, qd + kd);
        defer _ = mlx.mlx_array_free(k_c);
        const v_c = try self.sliceAxis(qkv, 1, qd + kd, qd + 2 * kd);
        defer _ = mlx.mlx_array_free(v_c);
        const q_h = try self.toHeads(q_c, n, self.heads);
        defer _ = mlx.mlx_array_free(q_h);
        const k_h = try self.toHeads(k_c, n, self.kv_heads);
        defer _ = mlx.mlx_array_free(k_h);
        const v = try self.toHeads(v_c, n, self.kv_heads);
        defer _ = mlx.mlx_array_free(v);
        const q = try self.rope(q_h, cos, sin);
        defer _ = mlx.mlx_array_free(q);
        const k = try self.rope(k_h, cos, sin);
        defer _ = mlx.mlx_array_free(k);

        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(self.head_dim)));
        const none = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(none);
        var o = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o);

        if (blk.kind == -1) {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, "", none, .{ .ctx = null }, false, self.s));
        } else if (blk.sinks.ctx == null) {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, "array", masks.band, .{ .ctx = null }, false, self.s));
        } else {
            // Rows [0, top): band + per-head sink on key 0.
            var sk = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sk);
            try mlx.check(mlx.mlx_astype(&sk, blk.sinks, mlx.mlx_array_dtype(q), self.s));
            var sk4 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sk4);
            const s4 = [_]c_int{ 1, @intCast(self.heads), 1, 1 };
            try mlx.check(mlx.mlx_reshape(&sk4, sk, &s4, 4, self.s));
            var sink_bias = mlx.mlx_array_new(); // [1,H,1,N]
            defer _ = mlx.mlx_array_free(sink_bias);
            try mlx.check(mlx.mlx_multiply(&sink_bias, sk4, masks.key0, self.s));
            var mask_top = mlx.mlx_array_new(); // [1,H,T,N]
            defer _ = mlx.mlx_array_free(mask_top);
            try mlx.check(mlx.mlx_add(&mask_top, masks.band_top, sink_bias, self.s));
            const q_top = try self.sliceAxis(q, 2, 0, masks.top);
            defer _ = mlx.mlx_array_free(q_top);
            var o_top = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(o_top);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o_top, q_top, k, v, scale, "array", mask_top, .{ .ctx = null }, false, self.s));
            if (masks.band_rest.ctx == null) {
                try mlx.check(mlx.mlx_array_set(&o, o_top));
            } else {
                const q_rest = try self.sliceAxis(q, 2, masks.top, n);
                defer _ = mlx.mlx_array_free(q_rest);
                var o_rest = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(o_rest);
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o_rest, q_rest, k, v, scale, "array", masks.band_rest, .{ .ctx = null }, false, self.s));
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                _ = mlx.mlx_vector_array_append_value(vec, o_top);
                _ = mlx.mlx_vector_array_append_value(vec, o_rest);
                try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 2, self.s));
            }
        }

        var ot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ot);
        const perm = [_]c_int{ 0, 2, 1, 3 };
        try mlx.check(mlx.mlx_transpose_axes(&ot, o, &perm, 4, self.s));
        var of = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(of);
        const fs = [_]c_int{ n, qd };
        try mlx.check(mlx.mlx_reshape(&of, ot, &fs, 2, self.s));
        return self.linear(of, blk.proj_w, blk.proj_b);
    }

    fn mlp(self: *MimoVision, h: mlx.mlx_array, blk: *const Block) !mlx.mlx_array {
        const g = try self.linear(h, blk.gate_w, blk.gate_b);
        defer _ = mlx.mlx_array_free(g);
        const u = try self.linear(h, blk.up_w, blk.up_b);
        defer _ = mlx.mlx_array_free(u);
        var sg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sg);
        try mlx.check(mlx.mlx_sigmoid(&sg, g, self.s));
        var silu = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(silu);
        try mlx.check(mlx.mlx_multiply(&silu, g, sg, self.s));
        var gu = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gu);
        try mlx.check(mlx.mlx_multiply(&gu, silu, u, self.s));
        return self.linear(gu, blk.down_w, blk.down_b);
    }

    fn takeRows(self: *MimoVision, x: mlx.mlx_array, idx: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_take_axis(&out, x, idx, 0, self.s));
        return out;
    }

    const Tables = struct {
        row_cos: mlx.mlx_array,
        row_sin: mlx.mlx_array,
        col_cos: mlx.mlx_array,
        col_sin: mlx.mlx_array,
        to_col: mlx.mlx_array, // [N] int32 token permutation row -> col order
        to_row: mlx.mlx_array, // inverse

        fn deinit(t: *Tables) void {
            inline for (.{ "row_cos", "row_sin", "col_cos", "col_sin", "to_col", "to_row" }) |f| _ = mlx.mlx_array_free(@field(t, f));
        }
    };

    /// Host-built 2-D rotary tables in the processor's merge-block token
    /// order, plus the column-major reordering at merge-unit granularity
    /// (`get_window_index_1d(col=True)` + `apply_index`).
    fn buildTables(self: *MimoVision, gh: u32, gw: u32) !Tables {
        const a = self.allocator;
        const m = self.merge;
        const n: usize = @as(usize, gh) * gw;
        const d: usize = self.head_dim; // 64
        const dim: usize = d / 2; // rotary dim 32
        const nf: usize = dim / 2; // 16 frequencies
        const inv = try a.alloc(f64, nf);
        defer a.free(inv);
        for (inv, 0..) |*v, i| v.* = 1.0 / std.math.pow(f64, ROPE_THETA, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(dim)));

        const cos_row = try a.alloc(f32, n * d);
        defer a.free(cos_row);
        const sin_row = try a.alloc(f32, n * d);
        defer a.free(sin_row);
        // token t in merge-block order: (bh, bw, ih, iw) with h = bh*m+ih, w = bw*m+iw
        var t: usize = 0;
        const lh = gh / m;
        const lw = gw / m;
        var bh: u32 = 0;
        while (bh < lh) : (bh += 1) {
            var bw: u32 = 0;
            while (bw < lw) : (bw += 1) {
                var ih: u32 = 0;
                while (ih < m) : (ih += 1) {
                    var iw: u32 = 0;
                    while (iw < m) : (iw += 1) {
                        const hp: f64 = @floatFromInt(bh * m + ih);
                        const wp: f64 = @floatFromInt(bw * m + iw);
                        const base = t * d;
                        for (0..nf) |j| {
                            const ah = hp * inv[j];
                            const aw = wp * inv[j];
                            // rot = [h·inv (16), w·inv (16)], emb = cat(rot, rot)
                            for ([_]usize{ 0, dim }) |off| {
                                cos_row[base + off + j] = @floatCast(@cos(ah));
                                sin_row[base + off + j] = @floatCast(@sin(ah));
                                cos_row[base + off + nf + j] = @floatCast(@cos(aw));
                                sin_row[base + off + nf + j] = @floatCast(@sin(aw));
                            }
                        }
                        t += 1;
                    }
                }
            }
        }

        // Column order over merge units: unit (bh, bw) -> index bw*lh + bh.
        const unit: usize = @as(usize, m) * m;
        const to_col = try a.alloc(i32, n);
        defer a.free(to_col);
        const to_row = try a.alloc(i32, n);
        defer a.free(to_row);
        var p: usize = 0;
        var cw: u32 = 0;
        while (cw < lw) : (cw += 1) {
            var ch: u32 = 0;
            while (ch < lh) : (ch += 1) {
                const src_unit: usize = @as(usize, ch) * lw + cw;
                for (0..unit) |u| {
                    to_col[p] = @intCast(src_unit * unit + u);
                    p += 1;
                }
            }
        }
        for (to_col, 0..) |src, dst| to_row[@intCast(src)] = @intCast(dst);

        const cos_col = try a.alloc(f32, n * d);
        defer a.free(cos_col);
        const sin_col = try a.alloc(f32, n * d);
        defer a.free(sin_col);
        for (to_col, 0..) |src, dst| {
            const si: usize = @intCast(src);
            @memcpy(cos_col[dst * d .. (dst + 1) * d], cos_row[si * d .. (si + 1) * d]);
            @memcpy(sin_col[dst * d .. (dst + 1) * d], sin_row[si * d .. (si + 1) * d]);
        }

        const tshape = [_]c_int{ 1, 1, @intCast(n), @intCast(d) };
        const ishape = [_]c_int{@intCast(n)};
        return .{
            .row_cos = mlx.mlx_array_new_data(cos_row.ptr, &tshape, 4, .float32),
            .row_sin = mlx.mlx_array_new_data(sin_row.ptr, &tshape, 4, .float32),
            .col_cos = mlx.mlx_array_new_data(cos_col.ptr, &tshape, 4, .float32),
            .col_sin = mlx.mlx_array_new_data(sin_col.ptr, &tshape, 4, .float32),
            .to_col = mlx.mlx_array_new_data(to_col.ptr, &ishape, 1, .int32),
            .to_row = mlx.mlx_array_new_data(to_row.ptr, &ishape, 1, .int32),
        };
    }

    /// Encode one image (one temporal-patch group). `patches` is
    /// [grid_h·grid_w, C·tps·ps·ps] in the Qwen2-VL processor's merge-block
    /// order; returns [1, N/merge², out_hidden].
    pub fn forward(self: *MimoVision, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        const n: c_int = @intCast(grid_h * grid_w);
        const dt: mlx.mlx_dtype = mlx.mlx_array_dtype(self.patch_w);
        var xin = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xin);
        try mlx.check(mlx.mlx_astype(&xin, patches, dt, self.s));
        var x = try self.linear(xin, self.patch_w, null);

        var tables = try self.buildTables(grid_h, grid_w);
        defer tables.deinit();
        var masks = try self.buildMasks(n, dt);
        defer masks.deinit();

        var in_col = false;
        for (self.blocks) |*blk| {
            const want_col = blk.kind == 1;
            if (want_col != in_col) {
                const moved = try self.takeRows(x, if (want_col) tables.to_col else tables.to_row);
                _ = mlx.mlx_array_free(x);
                x = moved;
                in_col = want_col;
            }
            const cos = if (in_col) tables.col_cos else tables.row_cos;
            const sin = if (in_col) tables.col_sin else tables.row_sin;
            {
                var h = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(h);
                try mlx.check(mlx.mlx_fast_rms_norm(&h, x, blk.norm1, EPS, self.s));
                const a = try self.attention(h, blk, cos, sin, &masks, n);
                defer _ = mlx.mlx_array_free(a);
                var nx = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&nx, x, a, self.s));
                _ = mlx.mlx_array_free(x);
                x = nx;
            }
            {
                var h = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(h);
                try mlx.check(mlx.mlx_fast_rms_norm(&h, x, blk.norm2, EPS, self.s));
                const f = try self.mlp(h, blk);
                defer _ = mlx.mlx_array_free(f);
                var nx = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&nx, x, f, self.s));
                _ = mlx.mlx_array_free(x);
                x = nx;
            }
        }
        if (in_col) {
            const moved = try self.takeRows(x, tables.to_row);
            _ = mlx.mlx_array_free(x);
            x = moved;
        }

        // Merger: LayerNorm (weight only) → [N/m², hidden·m²] → fc1 → GELU(erf) → fc2.
        var ln = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ln);
        try mlx.check(mlx.mlx_fast_layer_norm(&ln, x, self.merger_ln, .{ .ctx = null }, EPS, self.s));
        _ = mlx.mlx_array_free(x);
        const m2: c_int = @intCast(self.merge * self.merge);
        const nm: c_int = @divExact(n, m2);
        var g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g);
        const gs = [_]c_int{ nm, @intCast(self.hidden * self.merge * self.merge) };
        try mlx.check(mlx.mlx_reshape(&g, ln, &gs, 2, self.s));
        const f1 = try self.linear(g, self.merger_fc1, null);
        defer _ = mlx.mlx_array_free(f1);
        const gelu = blk: {
            const inv_sqrt2 = try self.scalar(std.math.sqrt1_2, dt);
            defer _ = mlx.mlx_array_free(inv_sqrt2);
            const half = try self.scalar(0.5, dt);
            defer _ = mlx.mlx_array_free(half);
            const one = try self.scalar(1.0, dt);
            defer _ = mlx.mlx_array_free(one);
            var xs = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xs);
            try mlx.check(mlx.mlx_multiply(&xs, f1, inv_sqrt2, self.s));
            var e = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(e);
            try mlx.check(mlx.mlx_erf(&e, xs, self.s));
            var e1 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(e1);
            try mlx.check(mlx.mlx_add(&e1, e, one, self.s));
            var xh = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xh);
            try mlx.check(mlx.mlx_multiply(&xh, f1, half, self.s));
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&out, xh, e1, self.s));
            break :blk out;
        };
        defer _ = mlx.mlx_array_free(gelu);
        const f2 = try self.linear(gelu, self.merger_fc2, null);
        defer _ = mlx.mlx_array_free(f2);
        var out = mlx.mlx_array_new();
        const os = [_]c_int{ 1, nm, @intCast(self.out_hidden) };
        try mlx.check(mlx.mlx_reshape(&out, f2, &os, 3, self.s));
        return out;
    }

    /// Encode one VIDEO: `grid_t` temporal-patch groups packed contiguously.
    /// The reference's cu_seqlens break attention at every group edge, the
    /// window reordering is within-frame, and the rotary table is spatial —
    /// so each group is exactly one `forward`, concatenated on the token axis.
    pub fn forwardVideo(self: *MimoVision, patches: mlx.mlx_array, grid_t: u32, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        if (grid_t <= 1) return self.forward(patches, grid_h, grid_w);
        const per: c_int = @intCast(grid_h * grid_w);
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var parts = try self.allocator.alloc(mlx.mlx_array, grid_t);
        defer {
            for (parts) |p| _ = mlx.mlx_array_free(p);
            self.allocator.free(parts);
        }
        for (0..grid_t) |ti| {
            const lo: c_int = @as(c_int, @intCast(ti)) * per;
            const grp = try self.sliceAxis(patches, 0, lo, lo + per);
            defer _ = mlx.mlx_array_free(grp);
            parts[ti] = try self.forward(grp, grid_h, grid_w);
            _ = mlx.mlx_vector_array_append_value(vec, parts[ti]);
        }
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, self.s));
        return out;
    }
};
