pub mod tokenizer;
pub mod model;
pub mod engine;

use core::ffi::{c_char, c_int, c_void};

#[allow(improper_ctypes)]
extern "C" {
    pub fn spp_load(path: *const c_char) -> *mut c_void;
    pub fn spp_free(ptr: *mut c_void);

    pub fn spp_encode_ids(
        ptr: *mut c_void,
        text: *const c_char,
        out_ids: *mut c_int,
        max_len: c_int,
    ) -> c_int;

    pub fn spp_decode_ids(
        ptr: *mut c_void,
        ids: *const c_int,
        len: c_int,
        out_buf: *mut c_char,
        buf_len: c_int,
    ) -> c_int;

    pub fn spp_vocab_size(ptr: *mut c_void) -> c_int;
}

pub mod llm_engine;
mod entropy;
