pub mod ffi;

use ffi::*;
use std::ffi::{CString, CStr};
use std::os::raw::{c_char, c_int, c_void};

pub struct Tokenizer {
    handle: *mut c_void,
}

impl Tokenizer {
    pub fn load(path: &str) -> Result<Self, String> {
        let cpath = CString::new(path).unwrap();
        let handle = unsafe { spp_load(cpath.as_ptr()) };
        if handle.is_null() {
            return Err("Failed to load SentencePiece model".into());
        }
        Ok(Self { handle })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i32>, String> {
        let ctext = CString::new(text).unwrap();

        let mut out = vec![0i32; 4096];

        let n = unsafe {
            spp_encode_ids(
                self.handle,
                ctext.as_ptr(),
                out.as_mut_ptr(),
                out.len() as c_int,
            )
        };

        if n < 0 {
            return Err("encode failed".into());
        }

        out.truncate(n as usize);
        Ok(out)
    }

    pub fn decode(&self, ids: &[i32]) -> Result<String, String> {
        let mut buf = vec![0i8; 8192];

        let n = unsafe {
            spp_decode_ids(
                self.handle,
                ids.as_ptr(),
                ids.len() as c_int,
                buf.as_mut_ptr(),
                buf.len() as c_int,
            )
        };

        if n < 0 {
            return Err("decode failed".into());
        }

        let s = unsafe {
            CStr::from_ptr(buf.as_ptr() as *const c_char)
                .to_string_lossy()
                .into_owned()
        };

        Ok(s)
    }

    pub fn vocab_size(&self) -> i32 {
        unsafe { spp_vocab_size(self.handle) }
    }
}

impl Drop for Tokenizer {
    fn drop(&mut self) {
        unsafe { spp_free(self.handle) }
    }
}

pub mod vocab;
