use std::fs::File;
use std::io::Read;

/// Pulls one byte of entropy from the webcam entropy pipe.
/// Falls back to pseudorandom if the pipe isn't available.
pub fn get_entropy_byte() -> u8 {
    if let Ok(mut file) = File::open("/tmp/unhidra_entropy.pipe") {
        let mut buf = [0u8; 1];
        if file.read_exact(&mut buf).is_ok() {
            return buf[0];
        }
    }
    rand::random::<u8>()
}
