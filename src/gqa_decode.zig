//! Single-token decode attention for grouped-query heads: the G query heads sharing a
//! KV head run as one R x D matrix (R = G padded to 8) on simdgroup MMA, so each K/V row
//! is read once and scored for the whole group; the fused vector kernel scores it once
//! per query head and is arithmetic bound from G ~ 4. Pass 1: one threadgroup per
//! (KV head, key chunk), four simdgroups with their own online softmax over 8-key
//! slabs, folded into one (max, sum, O) partial. Pass 2 merges the chunks.
const std = @import("std");
const mlx = @import("mlx.zig");

const HEADER = "#include <metal_simdgroup_matrix>\n";

const P1_SOURCE =
    \\    constexpr int RT = R / 8;              // row tiles
    \\    constexpr int CT = DV / 8;             // value column tiles
    \\    constexpr int NT = NSG * 32;           // threads
    \\    const int n_keys = params[0];
    \\    const int chunk = params[1];
    \\    const float scale = scalef[0];
    \\
    \\    const uint blk = threadgroup_position_in_grid.x;
    \\    const uint h = threadgroup_position_in_grid.y;
    \\    const uint sg = simdgroup_index_in_threadgroup;
    \\    const uint lane = thread_index_in_simdgroup;
    \\    const uint tid = thread_index_in_threadgroup;
    \\
    \\    // Q rows (T); after the key loop the fold reuses the same memory for O (float).
    \\    threadgroup float4 raw[SMEM16];
    \\    threadgroup T* qs = (threadgroup T*)raw;
    \\    threadgroup float* obuf = (threadgroup float*)raw;
    \\    threadgroup float red_m[NSG * R];
    \\    threadgroup float red_l[NSG * R];
    \\
    \\    const int q_hs = q_strides[1], q_ps = q_strides[2];
    \\    for (int i = tid; i < R * DK; i += NT) {
    \\        int r = i / DK, d = i % DK;
    \\        T val = T(0);
    \\        if (r < G * QL) {
    \\            int g = r % G, qp = r / G;
    \\            val = q[(h * G + g) * q_hs + qp * q_ps + d];
    \\        }
    \\        qs[i] = val;
    \\    }
    \\
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    const short qid = lane / 4;
    \\    const short fm = (qid & 4) + ((lane / 2) % 4);
    \\    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    \\
    \\    const long k_hs = k_strides[1], k_ss = k_strides[2];
    \\    const long v_hs = v_strides[1], v_ss = v_strides[2];
    \\    const device T* kh = k + h * k_hs;
    \\    const device T* vh = v + h * v_hs;
    \\
    \\    simdgroup_float8x8 O[RT][CT];
    \\    _Pragma("clang loop unroll(full)")
    \\    for (int a = 0; a < RT; a++)
    \\        _Pragma("clang loop unroll(full)")
    \\        for (int b = 0; b < CT; b++) O[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\    float m_run[RT], l_run[RT];
    \\    _Pragma("clang loop unroll(full)")
    \\    for (int a = 0; a < RT; a++) { m_run[a] = -INFINITY; l_run[a] = 0.0f; }
    \\
    \\    const int c0 = blk * chunk;
    \\    const int c1 = min(n_keys, c0 + chunk);
    \\    for (int kb0 = c0; kb0 < c1; kb0 += BK) {
    \\        const int ks = sg * 8;                       // this simdgroup's 8 keys in the block
    \\        if (kb0 + ks >= c1) continue;
    \\        const int kA = kb0 + ks + fn, kB = kA + 1, kV = kb0 + ks + fm;
    \\        simdgroup_float8x8 S[RT];
    \\        _Pragma("clang loop unroll(full)")
    \\        for (int a = 0; a < RT; a++) S[a] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\        _Pragma("clang loop unroll(full)")
    \\        for (int d0 = 0; d0 < DK; d0 += 8) {
    \\            simdgroup_float8x8 Bm;
    \\            thread auto& be = Bm.thread_elements();
    \\            be[0] = kA < c1 ? float(kh[kA * k_ss + d0 + fm]) : 0.0f;
    \\            be[1] = kB < c1 ? float(kh[kB * k_ss + d0 + fm]) : 0.0f;
    \\            _Pragma("clang loop unroll(full)")
    \\            for (int a = 0; a < RT; a++) {
    \\                simdgroup_float8x8 Am;
    \\                thread auto& ae = Am.thread_elements();
    \\                ae[0] = float(qs[(a * 8 + fm) * DK + d0 + fn]);
    \\                ae[1] = float(qs[(a * 8 + fm) * DK + d0 + fn + 1]);
    \\                simdgroup_multiply_accumulate(S[a], Am, Bm, S[a]);
    \\            }
    \\        }
    \\        const bool va = kb0 + ks + fn < c1, vb_ = kb0 + ks + fn + 1 < c1;
    \\        _Pragma("clang loop unroll(full)")
    \\        for (int a = 0; a < RT; a++) {
    \\            thread auto& se = S[a].thread_elements();
    \\            float s0 = va ? se[0] * scale : -INFINITY;
    \\            float s1 = vb_ ? se[1] * scale : -INFINITY;
    \\            float mx_ = max(s0, s1);
    \\            mx_ = max(mx_, simd_shuffle_xor(mx_, 1));
    \\            mx_ = max(mx_, simd_shuffle_xor(mx_, 8));
    \\            float m_new = max(m_run[a], mx_);
    \\            float alpha = (m_run[a] == -INFINITY) ? 0.0f : fast::exp(m_run[a] - m_new);
    \\            float p0 = (s0 == -INFINITY) ? 0.0f : fast::exp(s0 - m_new);
    \\            float p1 = (s1 == -INFINITY) ? 0.0f : fast::exp(s1 - m_new);
    \\            float ps = p0 + p1;
    \\            ps += simd_shuffle_xor(ps, 1);
    \\            ps += simd_shuffle_xor(ps, 8);
    \\            l_run[a] = l_run[a] * alpha + ps;
    \\            m_run[a] = m_new;
    \\            se[0] = p0; se[1] = p1;
    \\            _Pragma("clang loop unroll(full)")
    \\            for (int b = 0; b < CT; b++) {
    \\                thread auto& oe = O[a][b].thread_elements();
    \\                oe[0] *= alpha; oe[1] *= alpha;
    \\            }
    \\        }
    \\        _Pragma("clang loop unroll(full)")
    \\        for (int b = 0; b < CT; b++) {
    \\            simdgroup_float8x8 Vm;
    \\            thread auto& ve = Vm.thread_elements();
    \\            ve[0] = kV < c1 ? float(vh[kV * v_ss + b * 8 + fn]) : 0.0f;
    \\            ve[1] = kV < c1 ? float(vh[kV * v_ss + b * 8 + fn + 1]) : 0.0f;
    \\            _Pragma("clang loop unroll(full)")
    \\            for (int a = 0; a < RT; a++) simdgroup_multiply_accumulate(O[a][b], S[a], Vm, O[a][b]);
    \\        }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);   // Q region is now free for the fold
    \\
    \\    if ((lane & 9) == 0) {
    \\        _Pragma("clang loop unroll(full)")
    \\        for (int a = 0; a < RT; a++) {
    \\            red_m[sg * R + a * 8 + fm] = m_run[a];
    \\            red_l[sg * R + a * 8 + fm] = l_run[a];
    \\        }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    float f[RT];
    \\    _Pragma("clang loop unroll(full)")
    \\    for (int a = 0; a < RT; a++) {
    \\        int row = a * 8 + fm;
    \\        float mt = -INFINITY;
    \\        _Pragma("clang loop unroll(full)")
    \\        for (int s2 = 0; s2 < NSG; s2++) mt = max(mt, red_m[s2 * R + row]);
    \\        f[a] = (m_run[a] == -INFINITY) ? 0.0f : fast::exp(m_run[a] - mt);
    \\    }
    \\    _Pragma("clang loop unroll(full)")
    \\    for (int s2 = 0; s2 < NSG; s2++) {
    \\        if ((int)sg == s2) {
    \\            _Pragma("clang loop unroll(full)")
    \\            for (int a = 0; a < RT; a++)
    \\                _Pragma("clang loop unroll(full)")
    \\                for (int b = 0; b < CT; b++) {
    \\                    simdgroup_float8x8 acc = O[a][b];
    \\                    thread auto& ae = acc.thread_elements();
    \\                    ae[0] *= f[a]; ae[1] *= f[a];
    \\                    threadgroup float* dst = obuf + (a * 8) * DV + b * 8;
    \\                    if (s2 > 0) {
    \\                        simdgroup_float8x8 prev;
    \\                        simdgroup_load(prev, dst, DV);
    \\                        thread auto& pe = prev.thread_elements();
    \\                        ae[0] += pe[0]; ae[1] += pe[1];
    \\                    }
    \\                    simdgroup_store(acc, dst, DV);
    \\                }
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    const long base = ((long)blk * n_kv + h) * R;
    \\    for (int i = tid; i < R * DV; i += NT) o_part[base * DV + i] = obuf[i];
    \\    if (tid < R) {
    \\        float mt = -INFINITY;
    \\        _Pragma("clang loop unroll(full)")
    \\        for (int s2 = 0; s2 < NSG; s2++) mt = max(mt, red_m[s2 * R + tid]);
    \\        float lt = 0.0f;
    \\        _Pragma("clang loop unroll(full)")
    \\        for (int s2 = 0; s2 < NSG; s2++) {
    \\            float ms = red_m[s2 * R + tid];
    \\            if (ms != -INFINITY) lt += red_l[s2 * R + tid] * fast::exp(ms - mt);
    \\        }
    \\        ml_part[(base + tid) * 2 + 0] = mt;
    \\        ml_part[(base + tid) * 2 + 1] = lt;
    \\    }
;

const P2_SOURCE =
    \\    const uint d = thread_position_in_grid.x;       // value dim
    \\    const uint row = thread_position_in_grid.y;     // qpos * G + g
    \\    const uint h = thread_position_in_grid.z;       // kv head
    \\    const int nblk = params[2];
    \\    float mt = -INFINITY;
    \\    for (int b = 0; b < nblk; b++) mt = max(mt, ml_part[(((long)b * n_kv + h) * R + row) * 2]);
    \\    float acc = 0.0f, lt = 0.0f;
    \\    for (int b = 0; b < nblk; b++) {
    \\        long idx = ((long)b * n_kv + h) * R + row;
    \\        float mb = ml_part[idx * 2];
    \\        if (mb == -INFINITY) continue;
    \\        float w = fast::exp(mb - mt);
    \\        acc += w * o_part[idx * DV + d];
    \\        lt += w * ml_part[idx * 2 + 1];
    \\    }
    \\    const int g = row % G, qp = row / G;
    \\    out[((h * G + g) * QL + qp) * DV + d] = T(acc / lt);
;

/// Below this many keys the fused kernel is as fast (M3 Ultra, G 4..16).
pub const MIN_KEYS: c_int = 2048;
const NSG: c_int = 4;

var k1_cache: ?mlx.mlx_fast_metal_kernel = null;
var k2_cache: ?mlx.mlx_fast_metal_kernel = null;
var env_enabled: ?bool = null;
pub var enabled_override: ?bool = null; // test seam

/// MLX_SERVE_GQA_DECODE=0 keeps every decode on the fused kernel.
pub fn enabled() bool {
    if (enabled_override) |v| return v;
    if (env_enabled == null) {
        const raw = std.c.getenv("MLX_SERVE_GQA_DECODE");
        env_enabled = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    }
    return env_enabled.?;
}

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    // Strided inputs: the K/V views are slices of the cache's capacity buffer.
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn addInts(c: mlx.mlx_fast_metal_kernel_config, kvs: []const struct { [*:0]const u8, c_int }) !void {
    for (kvs) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, kv[0], kv[1]));
}

/// q [1,Hq,1,DK], k [1,Hkv,N,DK], v [1,Hkv,N,DV] (unit last-dim stride, views allowed),
/// no mask, no sinks -> [1,Hq,1,DV]. Null outside the kernel's envelope: the caller
/// runs the fused SDPA.
pub fn attend(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!enabled()) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    if (qs.len != 4 or ks.len != 4 or vs.len != 4) return null;
    if (qs[0] != 1 or ks[0] != 1 or vs[0] != 1 or qs[2] != 1) return null;
    const hq = qs[1];
    const hkv = ks[1];
    const n = ks[2];
    const dk = qs[3];
    const dv = vs[3];
    if (ks[3] != dk or vs[1] != hkv or vs[2] != n or hkv <= 0 or @rem(hq, hkv) != 0) return null;
    const g = @divExact(hq, hkv);
    if (g < 4 or g > 16 or n < MIN_KEYS) return null;
    if (@rem(dk, 8) != 0 or @rem(dv, 8) != 0 or dk > 256 or dv > 256) return null;
    const dt = mlx.mlx_array_dtype(q);
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(k) != dt or mlx.mlx_array_dtype(v) != dt) return null;
    for ([_]mlx.mlx_array{ q, k, v }) |a| if (mlx.mlx_array_strides(a)[3] != 1) return null;

    if (k1_cache == null) k1_cache = try makeKernel("msv_gqa_decode_p1", &.{ "q", "k", "v", "params", "scalef" }, &.{ "o_part", "ml_part" }, P1_SOURCE, HEADER);
    if (k2_cache == null) k2_cache = try makeKernel("msv_gqa_decode_p2", &.{ "o_part", "ml_part", "params" }, &.{"out"}, P2_SOURCE, "");

    const r: c_int = @divTrunc(g + 7, 8) * 8;
    const smem16: c_int = @divTrunc(@max(r * dk * 2, r * dv * 4) + 15, 16);
    const chunk: c_int = if (n < 65536) 512 else 1024;
    const nblk: c_int = @divTrunc(n + chunk - 1, chunk);

    const pdata = [_]i32{ n, chunk, nblk };
    const pshape = [_]c_int{3};
    const params = mlx.mlx_array_new_data(&pdata, &pshape, 1, .int32);
    defer _ = mlx.mlx_array_free(params);
    const sdata = [_]f32{scale};
    const sshape = [_]c_int{1};
    const scalef = mlx.mlx_array_new_data(&sdata, &sshape, 1, .float32);
    defer _ = mlx.mlx_array_free(scalef);

    const c1 = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c1);
    const op_shape = [_]c_int{ nblk, hkv, r, dv };
    const ml_shape = [_]c_int{ nblk, hkv, r, 2 };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c1, &op_shape, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c1, &ml_shape, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c1, 32 * NSG * nblk, hkv, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c1, 32 * NSG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c1, "T", dt));
    try addInts(c1, &.{ .{ "DK", dk }, .{ "DV", dv }, .{ "G", g }, .{ "QL", 1 }, .{ "R", r }, .{ "n_kv", hkv }, .{ "NSG", NSG }, .{ "BK", 8 * NSG }, .{ "SMEM16", smem16 } });

    const in1 = [_]mlx.mlx_array{ q, k, v, params, scalef };
    const v1 = mlx.mlx_vector_array_new_data(&in1, in1.len);
    defer _ = mlx.mlx_vector_array_free(v1);
    var o1 = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o1);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o1, k1_cache.?, v1, c1, s));
    var o_part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o_part);
    var ml_part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ml_part);
    try mlx.check(mlx.mlx_vector_array_get(&o_part, o1, 0));
    try mlx.check(mlx.mlx_vector_array_get(&ml_part, o1, 1));

    const c2 = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c2);
    const out_shape = [_]c_int{ 1, hq, 1, dv };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c2, &out_shape, 4, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c2, dv, g, hkv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c2, dv, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c2, "T", dt));
    try addInts(c2, &.{ .{ "DV", dv }, .{ "G", g }, .{ "QL", 1 }, .{ "R", r }, .{ "n_kv", hkv } });

    const in2 = [_]mlx.mlx_array{ o_part, ml_part, params };
    const v2 = mlx.mlx_vector_array_new_data(&in2, in2.len);
    defer _ = mlx.mlx_vector_array_free(v2);
    var o2 = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o2);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o2, k2_cache.?, v2, c2, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, o2, 0));
    return out;
}

fn testRand(rnd: std.Random, shape: []const c_int, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const data = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(data);
    for (data) |*x| x.* = (rnd.float(f32) - 0.5) * 4.0;
    const f = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, dt, s));
    return out;
}

fn testMaxDiff(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, a, b, s));
    var d32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d32);
    try mlx.check(mlx.mlx_astype(&d32, d, .float32, s));
    var ad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ad);
    try mlx.check(mlx.mlx_abs(&ad, d32, s));
    var m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_max(&m, ad, false, s));
    try mlx.check(mlx.mlx_array_eval(m));
    var out: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&out, m));
    return out;
}

test "gqa decode matches the fused SDPA on strided cache views (G 4, 12, 16; D 128/128, 192/128, 256/256)" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x6A0D);
    const rnd = prng.random();
    const cases = [_][5]c_int{ .{ 64, 4, 192, 128, 3001 }, .{ 32, 8, 128, 128, 5000 }, .{ 24, 2, 256, 256, 2048 } };
    for (cases) |c| {
        for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
            const cap = c[4] + 77;
            const q = try testRand(rnd, &.{ 1, c[0], 1, c[2] }, dt, s);
            defer _ = mlx.mlx_array_free(q);
            const kb = try testRand(rnd, &.{ 1, c[1], cap, c[2] }, dt, s);
            defer _ = mlx.mlx_array_free(kb);
            const vb = try testRand(rnd, &.{ 1, c[1], cap, c[3] }, dt, s);
            defer _ = mlx.mlx_array_free(vb);
            var k = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k);
            var v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v);
            const start = [_]c_int{ 0, 0, 0, 0 };
            const one = [_]c_int{ 1, 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&k, kb, &start, 4, &[_]c_int{ 1, c[1], c[4], c[2] }, 4, &one, 4, s));
            try mlx.check(mlx.mlx_slice(&v, vb, &start, 4, &[_]c_int{ 1, c[1], c[4], c[3] }, 4, &one, 4, s));
            const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c[2])));
            const got = (try attend(q, k, v, scale, s)) orelse return error.Declined;
            defer _ = mlx.mlx_array_free(got);
            var ref = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, q, k, v, scale, "", .{ .ctx = null }, .{ .ctx = null }, false, s));
            try std.testing.expect(try testMaxDiff(got, ref, s) < 4e-3);
        }
    }
}

test "gqa decode declines outside its envelope" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x6A0E);
    const rnd = prng.random();
    const q16 = try testRand(rnd, &.{ 1, 64, 1, 192 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(q16);
    const k_short = try testRand(rnd, &.{ 1, 4, MIN_KEYS - 1, 192 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(k_short);
    const v_short = try testRand(rnd, &.{ 1, 4, MIN_KEYS - 1, 128 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(v_short);
    try std.testing.expect((try attend(q16, k_short, v_short, 1.0, s)) == null);
    const q2 = try testRand(rnd, &.{ 1, 8, 1, 192 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(q2);
    const k = try testRand(rnd, &.{ 1, 4, MIN_KEYS, 192 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try testRand(rnd, &.{ 1, 4, MIN_KEYS, 128 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(v);
    try std.testing.expect((try attend(q2, k, v, 1.0, s)) == null); // G = 2: fused is at the read floor
    enabled_override = false;
    defer enabled_override = null;
    try std.testing.expect((try attend(q16, k, v, 1.0, s)) == null);
}
