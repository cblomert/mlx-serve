//! MiMo-V2.6 audio input: waveform → trunk-space soft tokens.
//!
//! Spec: vLLM's mimo_audio.py / processing_mimo.py (MiMo-Audio-Tokenizer +
//! MimoAudioEncoder), validated first as an MLX prototype
//! (scratchpad/mimo_audio_proto.py) that transcribes synthesized speech
//! word for word through the trunk. This file is that prototype in the C API.
//!
//!   pcm (24 kHz mono)
//!     → log-mel [T, 128]   (torchaudio MelSpectrogram: n_fft 960, hop 240,
//!                           hann, reflect-centred, HTK mel, power 1, log clip 1e-7)
//!     → tokenizer encoder, per 6000-frame segment:
//!         conv1 k3 + GELU, conv2 k3 s2 + GELU,
//!         24 pre-LN causal layers (even layers: left window 128), rotary 64,
//!         skip = output of layer 2 added back before the final LayerNorm,
//!         down-sample conv k2 s2 + GELU, LayerNorm
//!     → 20-level residual VQ (f32 nearest codeword)          [T', 20] codes
//!     → sum of 20 speech embeddings over groups of 4 codes   [S, 4, 1024]
//!     → 6-layer bidirectional Qwen2 per group (+ final norm)
//!     → [S, 4096] → Linear 16384 → GELU → Linear 4096        [1, S, 4096]
//!
//! 25 codes/s, 6.25 trunk tokens/s. Framing (server side):
//! <|mimo_audio_start|> <|audio_pad|>×S <|mimo_audio_end|>.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const Weights = model_mod.Weights;
const wav = @import("wav.zig");

pub const SAMPLE_RATE: u32 = 24000;
/// Rate assumed for headerless `mlx_pcm_f32` payloads (the server's existing
/// raw-PCM input_audio convention).
pub const RAW_PCM_RATE: u32 = 16000;

/// An `input_audio` payload → 24 kHz mono f32. Accepts a RIFF/WAVE file
/// (16/24-bit PCM or float, any rate/channel count) or headerless float32-LE
/// mono PCM at 16 kHz. Caller owns the result.
pub fn pcmFromPayload(allocator: std.mem.Allocator, bytes: []const u8) ![]f32 {
    if (bytes.len >= 12 and std.mem.eql(u8, bytes[0..4], "RIFF")) {
        const dec = try wav.decode(allocator, bytes);
        defer allocator.free(dec.pcm);
        const ch: usize = dec.channels;
        const frames = dec.pcm.len / ch;
        const mono = try allocator.alloc(f32, frames);
        defer allocator.free(mono);
        for (0..frames) |i| {
            var acc: f32 = 0;
            for (0..ch) |c| acc += dec.pcm[i * ch + c];
            mono[i] = acc / @as(f32, @floatFromInt(ch));
        }
        return wav.resampleLinear(allocator, mono, 1, dec.sample_rate, SAMPLE_RATE);
    }
    if (bytes.len % 4 != 0) return error.BadAudioPayload;
    const n = bytes.len / 4;
    const raw = try allocator.alloc(f32, n);
    defer allocator.free(raw);
    for (0..n) |i| raw[i] = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
    return wav.resampleLinear(allocator, raw, 1, RAW_PCM_RATE, SAMPLE_RATE);
}
const N_FFT: usize = 960;
const HOP: usize = 240;
const N_MELS: usize = 128;
const N_FREQ: usize = N_FFT / 2 + 1;
const SEGMENT: usize = 6000; // mel frames per tokenizer segment
const N_Q: usize = 20;
const GROUP: usize = 4;
const TOK_LAYERS: usize = 24;
const TOK_SKIP_AFTER: usize = 2; // encoder_skip_layer_id 3 → after layer index 2
const TOK_WINDOW: f32 = 128;
const LOCAL_LAYERS: usize = 6;
const HEADS: c_int = 16;
const HEAD_DIM: c_int = 64;
const D: c_int = 1024;

const TokLayer = struct {
    ln1_w: mlx.mlx_array,
    ln1_b: mlx.mlx_array,
    q_w: mlx.mlx_array,
    q_b: mlx.mlx_array,
    k_w: mlx.mlx_array,
    v_w: mlx.mlx_array,
    v_b: mlx.mlx_array,
    o_w: mlx.mlx_array,
    o_b: mlx.mlx_array,
    ln2_w: mlx.mlx_array,
    ln2_b: mlx.mlx_array,
    fc1_w: mlx.mlx_array,
    fc1_b: mlx.mlx_array,
    fc2_w: mlx.mlx_array,
    fc2_b: mlx.mlx_array,
};

const LocalLayer = struct {
    in_norm: mlx.mlx_array,
    post_norm: mlx.mlx_array,
    q_w: mlx.mlx_array,
    q_b: mlx.mlx_array,
    k_w: mlx.mlx_array,
    k_b: mlx.mlx_array,
    v_w: mlx.mlx_array,
    v_b: mlx.mlx_array,
    o_w: mlx.mlx_array,
    gate_w: mlx.mlx_array,
    up_w: mlx.mlx_array,
    down_w: mlx.mlx_array,
};

/// Owned temporaries of one computation step; freed together. MLX graph
/// nodes keep their inputs alive, so dropping a handle before eval is safe.
const Scope = struct {
    items: [64]mlx.mlx_array = undefined,
    n: usize = 0,

    fn add(sc: *Scope, a: mlx.mlx_array) mlx.mlx_array {
        std.debug.assert(sc.n < sc.items.len);
        sc.items[sc.n] = a;
        sc.n += 1;
        return a;
    }
    fn deinit(sc: *Scope) void {
        for (sc.items[0..sc.n]) |a| _ = mlx.mlx_array_free(a);
        sc.n = 0;
    }
};

pub const MimoAudio = struct {
    s: mlx.mlx_stream,
    allocator: std.mem.Allocator,
    tok_weights: Weights,
    enc_weights: Weights,

    dft_cos: mlx.mlx_array, // [N_FFT, N_FREQ] f32
    dft_sin: mlx.mlx_array,
    fbank: mlx.mlx_array, // [N_FREQ, N_MELS] f32
    window: []f32, // periodic hann

    conv1_w: mlx.mlx_array, // mlx layout [out, k, in]
    conv1_b: mlx.mlx_array,
    conv2_w: mlx.mlx_array,
    conv2_b: mlx.mlx_array,
    tok_layers: []TokLayer,
    tok_ln_w: mlx.mlx_array,
    tok_ln_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_ln_w: mlx.mlx_array,
    down_ln_b: mlx.mlx_array,
    codebooks: [N_Q]mlx.mlx_array, // [K, 1024] f32
    cb_sq: [N_Q]mlx.mlx_array, // [1, K] |E|²

    speech_emb: [N_Q]mlx.mlx_array, // [1280, 1024]
    local_layers: []LocalLayer,
    local_norm: mlx.mlx_array,
    proj0: mlx.mlx_array,
    proj2: mlx.mlx_array,

    pub fn initFromDir(allocator: std.mem.Allocator, model_dir: []const u8) !MimoAudio {
        var pb: [std.fs.max_path_bytes]u8 = undefined;
        const tok_path = try std.fmt.bufPrint(&pb, "{s}/audio_tokenizer/model.safetensors", .{model_dir});
        var tok_w = model_mod.loadWeightsSingleFile(allocator, tok_path) catch |err| {
            log.warn("[audio] mimo: cannot load {s} ({s})\n", .{ tok_path, @errorName(err) });
            return error.MissingAudioWeights;
        };
        errdefer tok_w.deinit();
        const enc_path = try std.fmt.bufPrint(&pb, "{s}/omnimodal/audio_encoder.safetensors", .{model_dir});
        var enc_w = model_mod.loadWeightsSingleFile(allocator, enc_path) catch |err| {
            log.warn("[audio] mimo: cannot load {s} ({s})\n", .{ enc_path, @errorName(err) });
            return error.MissingAudioWeights;
        };
        errdefer enc_w.deinit();

        const s = mlx.mlx_default_gpu_stream_new();
        var kb: [160]u8 = undefined;
        const get = struct {
            fn f(w: *const Weights, key: []const u8) !mlx.mlx_array {
                return w.get(key) orelse {
                    log.warn("[audio] mimo: missing {s}\n", .{key});
                    return error.MissingAudioWeights;
                };
            }
        }.f;
        const convT = struct {
            fn f(w: mlx.mlx_array, st: mlx.mlx_stream) !mlx.mlx_array {
                var t = mlx.mlx_array_new();
                const perm = [_]c_int{ 0, 2, 1 };
                try mlx.check(mlx.mlx_transpose_axes(&t, w, &perm, 3, st));
                return t;
            }
        }.f;

        var self: MimoAudio = undefined;
        self.s = s;
        self.allocator = allocator;

        self.conv1_w = try convT(try get(&tok_w, "encoder.conv1.weight"), s);
        self.conv1_b = try get(&tok_w, "encoder.conv1.bias");
        self.conv2_w = try convT(try get(&tok_w, "encoder.conv2.weight"), s);
        self.conv2_b = try get(&tok_w, "encoder.conv2.bias");
        self.down_w = try convT(try get(&tok_w, "encoder.down_sample_layer.0.weight"), s);
        self.down_ln_w = try get(&tok_w, "encoder.down_sample_norm.weight");
        self.down_ln_b = try get(&tok_w, "encoder.down_sample_norm.bias");
        self.tok_ln_w = try get(&tok_w, "encoder.layer_norm.weight");
        self.tok_ln_b = try get(&tok_w, "encoder.layer_norm.bias");

        self.tok_layers = try allocator.alloc(TokLayer, TOK_LAYERS);
        errdefer allocator.free(self.tok_layers);
        for (self.tok_layers, 0..) |*l, i| {
            const k = struct {
                fn f(buf: []u8, li: usize, rest: []const u8) []const u8 {
                    return std.fmt.bufPrint(buf, "encoder.layers.{d}.{s}", .{ li, rest }) catch unreachable;
                }
            }.f;
            l.* = .{
                .ln1_w = try get(&tok_w, k(&kb, i, "self_attn_layer_norm.weight")),
                .ln1_b = try get(&tok_w, k(&kb, i, "self_attn_layer_norm.bias")),
                .q_w = try get(&tok_w, k(&kb, i, "self_attn.q_proj.weight")),
                .q_b = try get(&tok_w, k(&kb, i, "self_attn.q_proj.bias")),
                .k_w = try get(&tok_w, k(&kb, i, "self_attn.k_proj.weight")),
                .v_w = try get(&tok_w, k(&kb, i, "self_attn.v_proj.weight")),
                .v_b = try get(&tok_w, k(&kb, i, "self_attn.v_proj.bias")),
                .o_w = try get(&tok_w, k(&kb, i, "self_attn.out_proj.weight")),
                .o_b = try get(&tok_w, k(&kb, i, "self_attn.out_proj.bias")),
                .ln2_w = try get(&tok_w, k(&kb, i, "final_layer_norm.weight")),
                .ln2_b = try get(&tok_w, k(&kb, i, "final_layer_norm.bias")),
                .fc1_w = try get(&tok_w, k(&kb, i, "fc1.weight")),
                .fc1_b = try get(&tok_w, k(&kb, i, "fc1.bias")),
                .fc2_w = try get(&tok_w, k(&kb, i, "fc2.weight")),
                .fc2_b = try get(&tok_w, k(&kb, i, "fc2.bias")),
            };
        }
        for (0..N_Q) |i| {
            const key = try std.fmt.bufPrint(&kb, "encoder.quantizer.vq.layers.{d}._codebook.embed", .{i});
            const e = try get(&tok_w, key);
            var ef = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&ef, e, .float32, s));
            self.codebooks[i] = ef;
            var sq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sq);
            try mlx.check(mlx.mlx_square(&sq, ef, s));
            var ss = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ss);
            try mlx.check(mlx.mlx_sum_axis(&ss, sq, 1, false, s));
            const kk: c_int = mlx.getShape(ef)[0];
            var r = mlx.mlx_array_new();
            const rs = [_]c_int{ 1, kk };
            try mlx.check(mlx.mlx_reshape(&r, ss, &rs, 2, s));
            self.cb_sq[i] = r;
        }

        for (0..N_Q) |i| {
            self.speech_emb[i] = try get(&enc_w, try std.fmt.bufPrint(&kb, "speech_embeddings.{d}.weight", .{i}));
        }
        self.local_layers = try allocator.alloc(LocalLayer, LOCAL_LAYERS);
        errdefer allocator.free(self.local_layers);
        for (self.local_layers, 0..) |*l, i| {
            const k = struct {
                fn f(buf: []u8, li: usize, rest: []const u8) []const u8 {
                    return std.fmt.bufPrint(buf, "audio_encoder.input_local_transformer.layers.{d}.{s}", .{ li, rest }) catch unreachable;
                }
            }.f;
            l.* = .{
                .in_norm = try get(&enc_w, k(&kb, i, "input_layernorm.weight")),
                .post_norm = try get(&enc_w, k(&kb, i, "post_attention_layernorm.weight")),
                .q_w = try get(&enc_w, k(&kb, i, "self_attn.q_proj.weight")),
                .q_b = try get(&enc_w, k(&kb, i, "self_attn.q_proj.bias")),
                .k_w = try get(&enc_w, k(&kb, i, "self_attn.k_proj.weight")),
                .k_b = try get(&enc_w, k(&kb, i, "self_attn.k_proj.bias")),
                .v_w = try get(&enc_w, k(&kb, i, "self_attn.v_proj.weight")),
                .v_b = try get(&enc_w, k(&kb, i, "self_attn.v_proj.bias")),
                .o_w = try get(&enc_w, k(&kb, i, "self_attn.o_proj.weight")),
                .gate_w = try get(&enc_w, k(&kb, i, "mlp.gate_proj.weight")),
                .up_w = try get(&enc_w, k(&kb, i, "mlp.up_proj.weight")),
                .down_w = try get(&enc_w, k(&kb, i, "mlp.down_proj.weight")),
            };
        }
        self.local_norm = try get(&enc_w, "audio_encoder.input_local_transformer.norm.weight");
        self.proj0 = try get(&enc_w, "audio_encoder.projection.mlp.0.weight");
        self.proj2 = try get(&enc_w, "audio_encoder.projection.mlp.2.weight");

        try self.buildFrontEnd();
        self.tok_weights = tok_w;
        self.enc_weights = enc_w;
        log.info("[audio] mimo audio encoder: tokenizer {d} layers, {d} codebooks, local {d} layers → {d}\n", .{
            TOK_LAYERS, N_Q, LOCAL_LAYERS, mlx.getShape(self.proj2)[0],
        });
        return self;
    }

    pub fn deinit(self: *MimoAudio) void {
        inline for (.{ "dft_cos", "dft_sin", "fbank", "conv1_w", "conv2_w", "down_w" }) |f| _ = mlx.mlx_array_free(@field(self, f));
        for (self.codebooks) |c| _ = mlx.mlx_array_free(c);
        for (self.cb_sq) |c| _ = mlx.mlx_array_free(c);
        self.allocator.free(self.window);
        self.allocator.free(self.tok_layers);
        self.allocator.free(self.local_layers);
        self.tok_weights.deinit();
        self.enc_weights.deinit();
    }

    /// DFT bases, HTK mel filterbank (torchaudio `melscale_fbanks`, norm None),
    /// periodic Hann window.
    fn buildFrontEnd(self: *MimoAudio) !void {
        const a = self.allocator;
        self.window = try a.alloc(f32, N_FFT);
        for (self.window, 0..) |*w, i| w.* = @floatCast(0.5 - 0.5 * @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(N_FFT))));

        const cs = try a.alloc(f32, N_FFT * N_FREQ);
        defer a.free(cs);
        const sn = try a.alloc(f32, N_FFT * N_FREQ);
        defer a.free(sn);
        for (0..N_FFT) |n| {
            for (0..N_FREQ) |k| {
                const ph = 2.0 * std.math.pi * @as(f64, @floatFromInt((n * k) % N_FFT)) / @as(f64, @floatFromInt(N_FFT));
                cs[n * N_FREQ + k] = @floatCast(@cos(ph));
                sn[n * N_FREQ + k] = @floatCast(@sin(ph));
            }
        }
        const bs = [_]c_int{ N_FFT, N_FREQ };
        self.dft_cos = mlx.mlx_array_new_data(cs.ptr, &bs, 2, .float32);
        self.dft_sin = mlx.mlx_array_new_data(sn.ptr, &bs, 2, .float32);

        const fb = try a.alloc(f32, N_FREQ * N_MELS);
        defer a.free(fb);
        const hz2mel = struct {
            fn f(hz: f64) f64 {
                return 2595.0 * std.math.log10(1.0 + hz / 700.0);
            }
        }.f;
        const mel2hz = struct {
            fn f(m: f64) f64 {
                return 700.0 * (std.math.pow(f64, 10.0, m / 2595.0) - 1.0);
            }
        }.f;
        var f_pts: [N_MELS + 2]f64 = undefined;
        const m_max = hz2mel(@as(f64, @floatFromInt(SAMPLE_RATE)) / 2.0);
        for (&f_pts, 0..) |*p, i| p.* = mel2hz(m_max * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(N_MELS + 1)));
        const nyq: f64 = @floatFromInt(SAMPLE_RATE / 2);
        for (0..N_FREQ) |fi| {
            const freq = nyq * @as(f64, @floatFromInt(fi)) / @as(f64, @floatFromInt(N_FREQ - 1));
            for (0..N_MELS) |m| {
                const down = (freq - f_pts[m]) / (f_pts[m + 1] - f_pts[m]);
                const up = (f_pts[m + 2] - freq) / (f_pts[m + 2] - f_pts[m + 1]);
                fb[fi * N_MELS + m] = @floatCast(@max(0.0, @min(down, up)));
            }
        }
        const fs = [_]c_int{ N_FREQ, N_MELS };
        self.fbank = mlx.mlx_array_new_data(fb.ptr, &fs, 2, .float32);
    }

    // ── op helpers (each returns a new owned array) ─────────────────────

    fn linear(self: *MimoAudio, x: mlx.mlx_array, w: mlx.mlx_array, b: ?mlx.mlx_array) !mlx.mlx_array {
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

    fn binop(self: *MimoAudio, comptime f: anytype, a: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(f(&out, a, b, self.s));
        return out;
    }

    fn reshape(self: *MimoAudio, x: mlx.mlx_array, shape: []const c_int) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out, x, shape.ptr, shape.len, self.s));
        return out;
    }

    fn transpose(self: *MimoAudio, x: mlx.mlx_array, perm: []const c_int) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_transpose_axes(&out, x, perm.ptr, perm.len, self.s));
        return out;
    }

    fn scalar(self: *MimoAudio, v: f32, dt: mlx.mlx_dtype) !mlx.mlx_array {
        const f = mlx.mlx_array_new_float(v);
        defer _ = mlx.mlx_array_free(f);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&out, f, dt, self.s));
        return out;
    }

    /// Exact (erf) GELU in the input dtype.
    fn gelu(self: *MimoAudio, x: mlx.mlx_array) !mlx.mlx_array {
        var sc = Scope{};
        defer sc.deinit();
        const dt = mlx.mlx_array_dtype(x);
        const inv = sc.add(try self.scalar(std.math.sqrt1_2, dt));
        const half = sc.add(try self.scalar(0.5, dt));
        const one = sc.add(try self.scalar(1.0, dt));
        const xs = sc.add(try self.binop(mlx.mlx_multiply, x, inv));
        var e = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_erf(&e, xs, self.s));
        _ = sc.add(e);
        const e1 = sc.add(try self.binop(mlx.mlx_add, e, one));
        const xh = sc.add(try self.binop(mlx.mlx_multiply, x, half));
        return self.binop(mlx.mlx_multiply, xh, e1);
    }

    fn layerNorm(self: *MimoAudio, x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_layer_norm(&out, x, w, b, 1e-5, self.s));
        return out;
    }

    fn rmsNorm(self: *MimoAudio, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, 1e-6, self.s));
        return out;
    }

    /// conv1d over [1, T, Cin] with an mlx-layout weight, + bias, + GELU.
    fn convGelu(self: *MimoAudio, x: mlx.mlx_array, w: mlx.mlx_array, b: ?mlx.mlx_array, stride: c_int, pad: c_int) !mlx.mlx_array {
        var y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_conv1d(&y, x, w, stride, pad, 1, 1, self.s));
        if (b) |bias| {
            const yb = try self.binop(mlx.mlx_add, y, bias);
            defer _ = mlx.mlx_array_free(yb);
            return self.gelu(yb);
        }
        return self.gelu(y);
    }

    /// Host rotary table [1, 1, n, 64] (default rope, rotate_half layout).
    fn ropeTable(self: *MimoAudio, n: usize, theta: f64) !struct { cos: mlx.mlx_array, sin: mlx.mlx_array } {
        const a = self.allocator;
        const d: usize = @intCast(HEAD_DIM);
        const c = try a.alloc(f32, n * d);
        defer a.free(c);
        const sn = try a.alloc(f32, n * d);
        defer a.free(sn);
        for (0..n) |p| {
            for (0..d / 2) |j| {
                const inv = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(d)));
                const ang = @as(f64, @floatFromInt(p)) * inv;
                c[p * d + j] = @floatCast(@cos(ang));
                c[p * d + j + d / 2] = @floatCast(@cos(ang));
                sn[p * d + j] = @floatCast(@sin(ang));
                sn[p * d + j + d / 2] = @floatCast(@sin(ang));
            }
        }
        const sh = [_]c_int{ 1, 1, @intCast(n), HEAD_DIM };
        return .{ .cos = mlx.mlx_array_new_data(c.ptr, &sh, 4, .float32), .sin = mlx.mlx_array_new_data(sn.ptr, &sh, 4, .float32) };
    }

    fn rope(self: *MimoAudio, x: mlx.mlx_array, cos: mlx.mlx_array, sin: mlx.mlx_array) !mlx.mlx_array {
        var sc = Scope{};
        defer sc.deinit();
        const dt = mlx.mlx_array_dtype(x);
        var xf = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&xf, x, .float32, self.s));
        _ = sc.add(xf);
        const sh = mlx.getShape(xf);
        const half: c_int = @divExact(HEAD_DIM, 2);
        var start: [4]c_int = .{ 0, 0, 0, 0 };
        var stop: [4]c_int = .{ sh[0], sh[1], sh[2], half };
        const strides: [4]c_int = .{ 1, 1, 1, 1 };
        var x1 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&x1, xf, &start, 4, &stop, 4, &strides, 4, self.s));
        _ = sc.add(x1);
        start[3] = half;
        stop[3] = HEAD_DIM;
        var x2 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&x2, xf, &start, 4, &stop, 4, &strides, 4, self.s));
        _ = sc.add(x2);
        var nx2 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_negative(&nx2, x2, self.s));
        _ = sc.add(nx2);
        var rot = mlx.mlx_array_new();
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, nx2);
            _ = mlx.mlx_vector_array_append_value(vec, x1);
            try mlx.check(mlx.mlx_concatenate_axis(&rot, vec, 3, self.s));
        }
        _ = sc.add(rot);
        const a = sc.add(try self.binop(mlx.mlx_multiply, xf, cos));
        const b = sc.add(try self.binop(mlx.mlx_multiply, rot, sin));
        const sum = sc.add(try self.binop(mlx.mlx_add, a, b));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&out, sum, dt, self.s));
        return out;
    }

    /// [B, n, H*64] → [B, H, n, 64]
    fn heads(self: *MimoAudio, x: mlx.mlx_array, b: c_int, n: c_int) !mlx.mlx_array {
        const r = try self.reshape(x, &.{ b, n, HEADS, HEAD_DIM });
        defer _ = mlx.mlx_array_free(r);
        return self.transpose(r, &.{ 0, 2, 1, 3 });
    }

    /// [B, H, n, 64] → [B, n, H*64]
    fn unheads(self: *MimoAudio, x: mlx.mlx_array, b: c_int, n: c_int) !mlx.mlx_array {
        const t = try self.transpose(x, &.{ 0, 2, 1, 3 });
        defer _ = mlx.mlx_array_free(t);
        return self.reshape(t, &.{ b, n, D });
    }

    // ── front end ───────────────────────────────────────────────────────

    /// log-mel [1, T, 128] f32 for 24 kHz mono samples.
    fn logMel(self: *MimoAudio, pcm: []const f32) !mlx.mlx_array {
        const a = self.allocator;
        const p = N_FFT / 2;
        // torch.stft(center=True, pad_mode="reflect") needs len > p.
        const len = pcm.len;
        const padded = try a.alloc(f32, len + 2 * p);
        defer a.free(padded);
        for (padded, 0..) |*v, i| {
            const j: isize = @as(isize, @intCast(i)) - @as(isize, @intCast(p));
            var k: isize = j;
            const L: isize = @intCast(len);
            if (k < 0) k = -k;
            if (k >= L) k = 2 * (L - 1) - k;
            v.* = pcm[@intCast(std.math.clamp(k, 0, L - 1))];
        }
        const n_frames = 1 + (padded.len - N_FFT) / HOP;
        const frames = try a.alloc(f32, n_frames * N_FFT);
        defer a.free(frames);
        for (0..n_frames) |f| {
            for (0..N_FFT) |i| frames[f * N_FFT + i] = padded[f * HOP + i] * self.window[i];
        }
        var sc = Scope{};
        defer sc.deinit();
        const fs = [_]c_int{ @intCast(n_frames), N_FFT };
        const fr = sc.add(mlx.mlx_array_new_data(frames.ptr, &fs, 2, .float32));
        const re = sc.add(try self.binop(mlx.mlx_matmul, fr, self.dft_cos));
        const im = sc.add(try self.binop(mlx.mlx_matmul, fr, self.dft_sin));
        const re2 = sc.add(try self.binop(mlx.mlx_multiply, re, re));
        const im2 = sc.add(try self.binop(mlx.mlx_multiply, im, im));
        const pw = sc.add(try self.binop(mlx.mlx_add, re2, im2));
        var mag = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sqrt(&mag, pw, self.s));
        _ = sc.add(mag);
        const mel = sc.add(try self.binop(mlx.mlx_matmul, mag, self.fbank));
        const floor_v = sc.add(mlx.mlx_array_new_float(1e-7));
        const cl = sc.add(try self.binop(mlx.mlx_maximum, mel, floor_v));
        var lg = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_log(&lg, cl, self.s));
        _ = sc.add(lg);
        return self.reshape(lg, &.{ 1, @intCast(n_frames), N_MELS });
    }

    // ── tokenizer encoder ───────────────────────────────────────────────

    fn tokAttention(self: *MimoAudio, h: mlx.mlx_array, l: *const TokLayer, n: c_int, cos: mlx.mlx_array, sin: mlx.mlx_array, swa: ?mlx.mlx_array) !mlx.mlx_array {
        var sc = Scope{};
        defer sc.deinit();
        const q0 = sc.add(try self.linear(h, l.q_w, l.q_b));
        const k0 = sc.add(try self.linear(h, l.k_w, null));
        const v0 = sc.add(try self.linear(h, l.v_w, l.v_b));
        const qh = sc.add(try self.heads(q0, 1, n));
        const kh = sc.add(try self.heads(k0, 1, n));
        const v = sc.add(try self.heads(v0, 1, n));
        const q = sc.add(try self.rope(qh, cos, sin));
        const k = sc.add(try self.rope(kh, cos, sin));
        var o = mlx.mlx_array_new();
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(HEAD_DIM)));
        if (swa) |m| {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, "array", m, .{ .ctx = null }, false, self.s));
        } else {
            const none = sc.add(mlx.mlx_array_new());
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, "causal", none, .{ .ctx = null }, false, self.s));
        }
        _ = sc.add(o);
        const of = sc.add(try self.unheads(o, 1, n));
        return self.linear(of, l.o_w, l.o_b);
    }

    /// Causal band mask: key j visible from query i iff i-128 <= j <= i.
    fn swaMask(self: *MimoAudio, n: c_int, dt: mlx.mlx_dtype) !mlx.mlx_array {
        var sc = Scope{};
        defer sc.deinit();
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_arange(&r, 0, @floatFromInt(n), 1, .float32, self.s));
        _ = sc.add(r);
        const rc = sc.add(try self.reshape(r, &.{ n, 1 }));
        const rr = sc.add(try self.reshape(r, &.{ 1, n }));
        const d = sc.add(try self.binop(mlx.mlx_subtract, rc, rr)); // i - j
        const zero = sc.add(mlx.mlx_array_new_float(0));
        const win = sc.add(mlx.mlx_array_new_float(TOK_WINDOW));
        const future = sc.add(try self.binop(mlx.mlx_less, d, zero));
        const far = sc.add(try self.binop(mlx.mlx_greater, d, win));
        const bad = sc.add(try self.binop(mlx.mlx_logical_or, future, far));
        const ninf = sc.add(mlx.mlx_array_new_float(-std.math.inf(f32)));
        var m = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&m, bad, ninf, zero, self.s));
        _ = sc.add(m);
        var md = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&md, m, dt, self.s));
        return md;
    }

    /// One ≤6000-frame mel segment [1, T, 128] → codes [T', 20] int32 (host).
    fn tokenizeSegment(self: *MimoAudio, mel: mlx.mlx_array, out: *std.ArrayList([N_Q]i32)) !void {
        const dt = mlx.mlx_array_dtype(self.conv1_b);
        var x: mlx.mlx_array = blk: {
            var mb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(mb);
            try mlx.check(mlx.mlx_astype(&mb, mel, dt, self.s));
            const c1 = try self.convGelu(mb, self.conv1_w, self.conv1_b, 1, 1);
            defer _ = mlx.mlx_array_free(c1);
            break :blk try self.convGelu(c1, self.conv2_w, self.conv2_b, 2, 1);
        };
        defer _ = mlx.mlx_array_free(x);
        const n: c_int = mlx.getShape(x)[1];

        const rt = try self.ropeTable(@intCast(n), 10000.0);
        defer _ = mlx.mlx_array_free(rt.cos);
        defer _ = mlx.mlx_array_free(rt.sin);
        const swa = try self.swaMask(n, dt);
        defer _ = mlx.mlx_array_free(swa);

        var skip: mlx.mlx_array = .{ .ctx = null };
        defer {
            if (skip.ctx != null) _ = mlx.mlx_array_free(skip);
        }
        for (self.tok_layers, 0..) |*l, i| {
            var sc = Scope{};
            defer sc.deinit();
            const h1 = sc.add(try self.layerNorm(x, l.ln1_w, l.ln1_b));
            const at = sc.add(try self.tokAttention(h1, l, n, rt.cos, rt.sin, if (i % 2 == 0) swa else null));
            const x1 = sc.add(try self.binop(mlx.mlx_add, x, at));
            const h2 = sc.add(try self.layerNorm(x1, l.ln2_w, l.ln2_b));
            const f1 = sc.add(try self.linear(h2, l.fc1_w, l.fc1_b));
            const g = sc.add(try self.gelu(f1));
            const f2 = sc.add(try self.linear(g, l.fc2_w, l.fc2_b));
            const nx = try self.binop(mlx.mlx_add, x1, f2);
            _ = mlx.mlx_array_free(x);
            x = nx;
            if (i == TOK_SKIP_AFTER) {
                var cp = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&cp, x));
                skip = cp;
            }
            // Bound the lazy graph: one layer at a time.
            try mlx.check(mlx.mlx_array_eval(x));
        }

        var sc = Scope{};
        defer sc.deinit();
        const xs = sc.add(try self.binop(mlx.mlx_add, x, skip));
        var hs = sc.add(try self.layerNorm(xs, self.tok_ln_w, self.tok_ln_b));
        if (@mod(n, 2) != 0) {
            const zs = [_]c_int{ 1, 1, D };
            var z = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&z, &zs, 3, dt, self.s));
            _ = sc.add(z);
            var cat = mlx.mlx_array_new();
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, hs);
            _ = mlx.mlx_vector_array_append_value(vec, z);
            try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 1, self.s));
            hs = sc.add(cat);
        }
        const ds = sc.add(try self.convGelu(hs, self.down_w, null, 2, 0));
        const dn = sc.add(try self.layerNorm(ds, self.down_ln_w, self.down_ln_b));
        const t2: c_int = mlx.getShape(dn)[1];
        const d2 = sc.add(try self.reshape(dn, &.{ t2, D }));
        var res = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&res, d2, .float32, self.s));

        // Residual VQ: nearest codeword by argmax(-(|x|² - 2x·E^T + |E|²)).
        const two = sc.add(mlx.mlx_array_new_float(2.0));
        const idx_arrs = try self.allocator.alloc(mlx.mlx_array, N_Q);
        defer self.allocator.free(idx_arrs);
        var n_idx: usize = 0;
        defer {
            for (idx_arrs[0..n_idx]) |ia| _ = mlx.mlx_array_free(ia);
        }
        defer _ = mlx.mlx_array_free(res);
        for (0..N_Q) |qi| {
            var lsc = Scope{};
            defer lsc.deinit();
            const et = lsc.add(try self.transpose(self.codebooks[qi], &.{ 1, 0 }));
            const xe = lsc.add(try self.binop(mlx.mlx_matmul, res, et));
            const xe2 = lsc.add(try self.binop(mlx.mlx_multiply, xe, two));
            // |x|² is constant per row — argmin(|E|² - 2x·E) is the same index.
            const dist = lsc.add(try self.binop(mlx.mlx_subtract, self.cb_sq[qi], xe2));
            const neg = lsc.add(blk: {
                var ng = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_negative(&ng, dist, self.s));
                break :blk ng;
            });
            var idx = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_argmax_axis(&idx, neg, 1, false, self.s));
            var idx32 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&idx32, idx, .int32, self.s));
            _ = mlx.mlx_array_free(idx);
            idx_arrs[n_idx] = idx32;
            n_idx += 1;
            var q = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_take_axis(&q, self.codebooks[qi], idx32, 0, self.s));
            _ = lsc.add(q);
            const nr = try self.binop(mlx.mlx_subtract, res, q);
            _ = mlx.mlx_array_free(res);
            res = nr;
        }
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (idx_arrs[0..n_idx]) |ia| _ = mlx.mlx_vector_array_append_value(vec, ia);
        try mlx.check(mlx.mlx_eval(vec));

        const tn: usize = @intCast(t2);
        const base = out.items.len;
        try out.resize(self.allocator, base + tn);
        for (idx_arrs[0..n_idx], 0..) |ia, qi| {
            const p = mlx.mlx_array_data_int32(ia) orelse return error.AudioEncodeFailed;
            for (0..tn) |t| out.items[base + t][qi] = p[t];
        }
    }

    // ── LLM-side encoder ────────────────────────────────────────────────

    fn localForward(self: *MimoAudio, codes: []const [N_Q]i32) !mlx.mlx_array {
        const a = self.allocator;
        const s_groups: usize = (codes.len + GROUP - 1) / GROUP;
        const padded_n = s_groups * GROUP;
        const col = try a.alloc(i32, padded_n);
        defer a.free(col);

        const sg: c_int = @intCast(s_groups);
        var x: mlx.mlx_array = .{ .ctx = null };
        for (0..N_Q) |qi| {
            for (0..padded_n) |t| col[t] = codes[@min(t, codes.len - 1)][qi]; // pad by repeating the last frame
            const is = [_]c_int{@intCast(padded_n)};
            const ia = mlx.mlx_array_new_data(col.ptr, &is, 1, .int32);
            defer _ = mlx.mlx_array_free(ia);
            var e = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_take_axis(&e, self.speech_emb[qi], ia, 0, self.s));
            if (x.ctx == null) {
                x = e;
            } else {
                const nx = try self.binop(mlx.mlx_add, x, e);
                _ = mlx.mlx_array_free(e);
                _ = mlx.mlx_array_free(x);
                x = nx;
            }
        }
        {
            const r = try self.reshape(x, &.{ sg, GROUP, D });
            _ = mlx.mlx_array_free(x);
            x = r;
        }

        const rt = try self.ropeTable(GROUP, 640000.0);
        defer _ = mlx.mlx_array_free(rt.cos);
        defer _ = mlx.mlx_array_free(rt.sin);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(HEAD_DIM)));
        const g: c_int = @intCast(GROUP);
        for (self.local_layers) |*l| {
            var sc = Scope{};
            defer sc.deinit();
            const h = sc.add(try self.rmsNorm(x, l.in_norm));
            const q0 = sc.add(try self.linear(h, l.q_w, l.q_b));
            const k0 = sc.add(try self.linear(h, l.k_w, l.k_b));
            const v0 = sc.add(try self.linear(h, l.v_w, l.v_b));
            const qh = sc.add(try self.heads(q0, sg, g));
            const kh = sc.add(try self.heads(k0, sg, g));
            const v = sc.add(try self.heads(v0, sg, g));
            const q = sc.add(try self.rope(qh, rt.cos, rt.sin));
            const k = sc.add(try self.rope(kh, rt.cos, rt.sin));
            const none = sc.add(mlx.mlx_array_new());
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, "", none, .{ .ctx = null }, false, self.s));
            _ = sc.add(o);
            const of = sc.add(try self.unheads(o, sg, g));
            const at = sc.add(try self.linear(of, l.o_w, null));
            const x1 = sc.add(try self.binop(mlx.mlx_add, x, at));
            const h2 = sc.add(try self.rmsNorm(x1, l.post_norm));
            const gt = sc.add(try self.linear(h2, l.gate_w, null));
            const up = sc.add(try self.linear(h2, l.up_w, null));
            var sg_ = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_sigmoid(&sg_, gt, self.s));
            _ = sc.add(sg_);
            const silu = sc.add(try self.binop(mlx.mlx_multiply, gt, sg_));
            const gu = sc.add(try self.binop(mlx.mlx_multiply, silu, up));
            const dn = sc.add(try self.linear(gu, l.down_w, null));
            const nx = try self.binop(mlx.mlx_add, x1, dn);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }
        var sc = Scope{};
        defer sc.deinit();
        _ = sc.add(x);
        const xn = sc.add(try self.rmsNorm(x, self.local_norm));
        const flat = sc.add(try self.reshape(xn, &.{ sg, g * D }));
        const p0 = sc.add(try self.linear(flat, self.proj0, null));
        const ge = sc.add(try self.gelu(p0));
        const p2 = sc.add(try self.linear(ge, self.proj2, null));
        const out_h: c_int = mlx.getShape(self.proj2)[0];
        return self.reshape(p2, &.{ 1, sg, out_h });
    }

    /// Encode one clip of 24 kHz mono samples → [1, S, out_hidden].
    pub fn forward(self: *MimoAudio, pcm: []const f32) !mlx.mlx_array {
        if (pcm.len < N_FFT) return error.AudioTooShort;
        const mel = try self.logMel(pcm);
        defer _ = mlx.mlx_array_free(mel);
        const t_mel: usize = @intCast(mlx.getShape(mel)[1]);

        var codes = std.ArrayList([N_Q]i32).empty;
        defer codes.deinit(self.allocator);
        var off: usize = 0;
        while (off < t_mel) : (off += SEGMENT) {
            const hi = @min(off + SEGMENT, t_mel);
            const start = [_]c_int{ 0, @intCast(off), 0 };
            const stop = [_]c_int{ 1, @intCast(hi), N_MELS };
            const strides = [_]c_int{ 1, 1, 1 };
            var seg = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(seg);
            try mlx.check(mlx.mlx_slice(&seg, mel, &start, 3, &stop, 3, &strides, 3, self.s));
            try self.tokenizeSegment(seg, &codes);
        }
        if (codes.items.len == 0) return error.AudioTooShort;
        const out = try self.localForward(codes.items);
        log.info("[audio] mimo: {d} samples → {d} mel frames → {d} codes → {d} tokens\n", .{ pcm.len, t_mel, codes.items.len, mlx.getShape(out)[1] });
        return out;
    }

    /// Soft-token count `forward` will produce for `n_samples` at 24 kHz
    /// (processing_mimo's token_len formula).
    pub fn tokenCount(n_samples: usize) usize {
        if (n_samples < N_FFT) return 0;
        const t_mel = 1 + n_samples / HOP; // centred STFT frame count
        var total: usize = 0;
        var off: usize = 0;
        while (off < t_mel) : (off += SEGMENT) {
            const seg = @min(SEGMENT, t_mel - off);
            const c2 = (seg - 1) / 2 + 1;
            total += (c2 + 1) / 2;
        }
        return (total + GROUP - 1) / GROUP;
    }
};

test "tokenCount matches processing_mimo's token_len formula" {
    // Reference: n = T + 3 - 3; n = (n + 2 - 3)//2 + 1; n = n//2 + (n%2 != 0); ceil(n/4),
    // with T = 1 + samples/240 (centred STFT), per 6000-frame segment.
    const cases = [_]struct { samples: usize, want: usize }{
        .{ .samples = 155299, .want = 41 }, // 6.47 s → 648 mel → 162 codes
        .{ .samples = 244783, .want = 64 }, // 1020 mel → 255 codes
        .{ .samples = 1958679, .want = 511 }, // 8162 mel, two segments → 2041 codes
        .{ .samples = 100, .want = 0 },
    };
    for (cases) |c| try std.testing.expectEqual(c.want, MimoAudio.tokenCount(c.samples));
}

test "pcmFromPayload: raw f32 PCM is read as 16 kHz and resampled to 24 kHz" {
    const a = std.testing.allocator;
    var raw: [1600]f32 = undefined;
    for (&raw, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 1600.0;
    const out = try pcmFromPayload(a, std.mem.sliceAsBytes(&raw));
    defer a.free(out);
    try std.testing.expectEqual(@as(usize, 2400), out.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[1200], 1e-3);
    try std.testing.expectError(error.BadAudioPayload, pcmFromPayload(a, "abc"));
}
