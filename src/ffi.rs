use libc::{c_int, c_float};

#[repr(C)]
pub struct tensor {
    pub rows: c_int,
    pub cols: c_int,
    pub data: *mut c_float,
}

#[repr(C)]
pub struct linear_layer {
    pub weight: *mut tensor,
    pub bias: *mut tensor,
    pub in_features: c_int,
    pub out_features: c_int,
}

#[repr(C)]
pub struct attention_head {
    pub Wq: *mut linear_layer,
    pub Wk: *mut linear_layer,
    pub Wv: *mut linear_layer,
    pub embed_dim: c_int,
}

#[repr(C)]
pub struct multi_head_attention {
    pub embed_dim: c_int,
    pub head_count: c_int,
    pub head_dim: c_int,
    pub heads: *mut *mut attention_head,
    pub out_proj: *mut linear_layer,
}

#[repr(C)]
pub struct mlp {
    pub layer1: *mut linear_layer,
    pub layer2: *mut linear_layer,
}

#[repr(C)]
pub struct layernorm {
    pub gamma: *mut tensor,
    pub beta: *mut tensor,
    pub size: c_int,
}

unsafe extern "C" {
    pub fn tensor_new(rows: c_int, cols: c_int) -> *mut tensor;
    pub fn tensor_free(t: *mut tensor);
    pub fn tensor_matmul(a: *const tensor, b: *const tensor, out: *mut tensor);
}

unsafe extern "C" {
    pub fn linear_new(in_features: c_int, out_features: c_int) -> *mut linear_layer;
    pub fn linear_free(layer: *mut linear_layer);
    pub fn linear_forward(layer: *const linear_layer, input: *const tensor, out: *mut tensor);
}

unsafe extern "C" {
    pub fn attention_new(embed_dim: c_int) -> *mut attention_head;
    pub fn attention_free(att: *mut attention_head);
    pub fn attention_forward(att: *const attention_head, x: *const tensor, out: *mut tensor);
}

unsafe extern "C" {
    pub fn mha_new(embed_dim: c_int, head_count: c_int) -> *mut multi_head_attention;
    pub fn mha_free(m: *mut multi_head_attention);
    pub fn mha_forward(m: *const multi_head_attention, x: *const tensor, out: *mut tensor);
}

unsafe extern "C" {
    pub fn mlp_new(input_size: c_int, hidden_size: c_int) -> *mut mlp;
    pub fn mlp_free(m: *mut mlp);
    pub fn mlp_forward(m: *const mlp, input: *const tensor, out: *mut tensor);
}

unsafe extern "C" {
    pub fn layernorm_new(size: c_int) -> *mut layernorm;
    pub fn layernorm_free(ln: *mut layernorm);
    pub fn layernorm_forward(ln: *const layernorm, input: *const tensor, out: *mut tensor);
}

unsafe extern "C" {
    pub fn softmax(t: *mut tensor);
}
