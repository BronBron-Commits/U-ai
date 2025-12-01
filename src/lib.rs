pub mod tokenizer;
pub mod llm_engine;
pub mod model;

#[allow(improper_ctypes)]
extern "C" {
    pub fn spp_load(path: *const i8) -> *mut core::ffi::c_void;
    pub fn spp_free(ptr: *mut core::ffi::c_void);

    pub fn spp_encode_ids(
        ptr: *mut core::ffi::c_void,
        text: *const i8,
        out_ids: *mut i32,
        max_len: i32,
    ) -> i32;

    pub fn spp_decode_ids(
        ptr: *mut i8,
        ids: *const i32,
        len: i32,
        out_buf: *mut i8,
        buf_len: i32,
    ) -> i32;

    pub fn spp_vocab_size(ptr: *mut core::ffi::c_void) -> i32;
}
pub mod dataset;
