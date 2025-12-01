use libc::{c_char, c_int};
use std::ffi::{CString, CStr};
use std::ptr;

#[link(name = "sentencepiece")]
extern "C" {
    fn sentencepiece_processor_new() -> *mut std::ffi::c_void;
    fn sentencepiece_processor_delete(sp: *mut std::ffi::c_void);
    fn sentencepiece_processor_load(
        sp: *mut std::ffi::c_void,
        model_path: *const c_char
    ) -> c_int;
    fn sentencepiece_processor_encode(
        sp: *mut std::ffi::c_void,
        input: *const c_char,
        out: *mut *mut c_int,
        out_size: *mut c_int
    ) -> c_int;
    fn sentencepiece_processor_decode(
        sp: *mut std::ffi::c_void,
        ids: *const c_int,
        size: c_int,
        out: *mut *mut c_char
    ) -> c_int;
}

pub struct Tokenizer {
    inner: *mut std::ffi::c_void,
}

impl Tokenizer {
    pub fn load(path: &str) -> Self {
        let cpath = CString::new(path).unwrap();
        unsafe {
            let sp = sentencepiece_processor_new();
            if sentencepiece_processor_load(sp, cpath.as_ptr()) != 0 {
                panic!("Failed to load SentencePiece model");
            }
            Self { inner: sp }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        let ctext = CString::new(text).unwrap();
        let mut out_ptr: *mut c_int = ptr::null_mut();
        let mut out_len: c_int = 0;

        unsafe {
            sentencepiece_processor_encode(
                self.inner,
                ctext.as_ptr(),
                &mut out_ptr,
                &mut out_len,
            );
            let slice = std::slice::from_raw_parts(out_ptr, out_len as usize);
            slice.to_vec()
        }
    }

    pub fn decode(&self, ids: &[i32]) -> String {
        let mut out_ptr: *mut c_char = ptr::null_mut();
        unsafe {
            sentencepiece_processor_decode(
                self.inner,
                ids.as_ptr(),
                ids.len() as c_int,
                &mut out_ptr
            );
            CStr::from_ptr(out_ptr).to_string_lossy().to_string()
        }
    }
}

impl Drop for Tokenizer {
    fn drop(&mut self) {
        unsafe {
            sentencepiece_processor_delete(self.inner);
        }
    }
}
