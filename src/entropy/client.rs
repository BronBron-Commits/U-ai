use std::time::Duration;

pub fn fetch_entropy_chunk(url: &str) -> Option<Vec<u8>> {
    let timeout = Duration::from_secs(2);

    // correct API use for ureq 2.x
    let resp = ureq::get(url)
        .timeout(timeout)
        .call()
        .ok()?;

    let mut buf = Vec::new();
    resp.into_reader().read_to_end(&mut buf).ok()?;

    Some(buf)
}
