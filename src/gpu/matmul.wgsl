@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;

    let width : u32 = 1024u; // configurable later
    var sum : f32 = 0.0;

    for (var k = 0u; k < width; k = k + 1u) {
        sum = sum + A[row * width + k] * B[k * width + col];
    }

    C[row * width + col] = sum;
}
