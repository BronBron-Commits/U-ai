use std::time::Duration;
use std::io::Read;

pub fn fetch_entropy_chunk(url: &str) -> Option<Vec<u8>> {
    let timeout = Duration::from_millis(1500);

    let resp = ureq::get(url)
        .timeout(timeout)
        .call()
        .ok()?;

    let mut buf = Vec::new();
    resp.into_reader().read_to_end(&mut buf).ok()?;

    Some(buf)
}
