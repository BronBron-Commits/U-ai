use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use anyhow::{Result, bail};

// Very small GGUF header parser
pub struct Gguf {
    pub tensors: Vec<TensorInfo>,
}

pub struct TensorInfo {
    pub name: String,
    pub offset: u64,
    pub n_elements: u64,
}

impl Gguf {
    pub fn load(path: &str) -> Result<Self> {
        let mut f = File::open(path)?;

        // read magic
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            bail!("Not a GGUF file");
        }

        // version (we accept anything)
        let mut version = [0u8; 4];
        f.read_exact(&mut version)?;

        // skip metadata counts 
        // (we will improve this when needed)
        let mut buf8 = [0u8; 8];

        // read number of tensors
        f.read_exact(&mut buf8)?;
        let n_tensors = u64::from_le_bytes(buf8);

        let mut tensors = Vec::new();

        for _ in 0..n_tensors {
            // read name length
            f.read_exact(&mut buf8)?;
            let name_len = u64::from_le_bytes(buf8);

            // read name
            let mut name_buf = vec![0u8; name_len as usize];
            f.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf)?;

            // read offset
            f.read_exact(&mut buf8)?;
            let offset = u64::from_le_bytes(buf8);

            // read number of elements
            f.read_exact(&mut buf8)?;
            let n_elements = u64::from_le_bytes(buf8);

            tensors.push(TensorInfo {
                name,
                offset,
                n_elements,
            });
        }

        Ok(Self { tensors })
    }
}
