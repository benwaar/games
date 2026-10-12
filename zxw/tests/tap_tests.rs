/// Integration tests for the TAP parser (src/tap.rs).
///
/// These tests work with the `rlib` crate target so they can import directly.
use zxw::tap::parse_tap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a raw TAP block: 2-byte LE length + flag + payload + XOR checksum.
fn make_tap_block(flag: u8, payload: &[u8]) -> Vec<u8> {
    let body_len = 1 + payload.len() + 1; // flag + payload + checksum
    let mut out = Vec::with_capacity(2 + body_len);
    out.push((body_len & 0xFF) as u8);
    out.push((body_len >> 8) as u8);
    out.push(flag);
    out.extend_from_slice(payload);
    let checksum = std::iter::once(flag)
        .chain(payload.iter().copied())
        .fold(0u8, |acc, b| acc ^ b);
    out.push(checksum);
    out
}

/// Build a 17-byte header payload.
fn header_payload(
    block_type: u8,
    name: &[u8; 10],
    data_len: u16,
    param1: u16,
    param2: u16,
) -> Vec<u8> {
    let mut p = vec![block_type];
    p.extend_from_slice(name);
    p.extend_from_slice(&data_len.to_le_bytes());
    p.extend_from_slice(&param1.to_le_bytes());
    p.extend_from_slice(&param2.to_le_bytes());
    p
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn parse_header_block() {
    let name = *b"MyProg    ";
    let hdr_payload = header_payload(3, &name, 256, 0x8000, 256);
    let tap = make_tap_block(0x00, &hdr_payload);

    let blocks = parse_tap(&tap).expect("parse should succeed");
    assert_eq!(blocks.len(), 1);

    let blk = &blocks[0];
    assert_eq!(blk.flag, 0x00);
    assert_eq!(blk.data, hdr_payload);

    let hdr = blk.header.as_ref().expect("header should be Some");
    assert_eq!(hdr.block_type, 3); // Code
    assert_eq!(&hdr.name, b"MyProg    ");
    assert_eq!(hdr.data_len, 256);
    assert_eq!(hdr.param1, 0x8000); // load address
}

#[test]
fn parse_data_block() {
    let data: Vec<u8> = (0u8..=15).collect();
    let tap = make_tap_block(0xFF, &data);

    let blocks = parse_tap(&tap).expect("parse should succeed");
    assert_eq!(blocks.len(), 1);

    let blk = &blocks[0];
    assert_eq!(blk.flag, 0xFF);
    assert_eq!(blk.data, data);
    assert!(blk.header.is_none());
}

#[test]
fn parse_header_plus_data() {
    let name = *b"TestCode  ";
    let payload_data: Vec<u8> = vec![0x3E, 0x42, 0x76]; // LD A,0x42; HALT
    let hdr = header_payload(3, &name, payload_data.len() as u16, 0x9000, payload_data.len() as u16);

    let mut tap = make_tap_block(0x00, &hdr);
    tap.extend(make_tap_block(0xFF, &payload_data));

    let blocks = parse_tap(&tap).expect("parse should succeed");
    assert_eq!(blocks.len(), 2);

    // Header
    assert_eq!(blocks[0].flag, 0x00);
    let h = blocks[0].header.as_ref().unwrap();
    assert_eq!(h.data_len, 3);
    assert_eq!(h.param1, 0x9000);

    // Data
    assert_eq!(blocks[1].flag, 0xFF);
    assert_eq!(blocks[1].data, payload_data);
}

#[test]
fn checksum_mismatch_is_an_error() {
    let mut tap = make_tap_block(0xFF, &[1, 2, 3]);
    // Flip the last byte (the stored checksum).
    let last = tap.len() - 1;
    tap[last] ^= 0xAA;

    assert!(parse_tap(&tap).is_err(), "bad checksum should error");
}

#[test]
fn truncated_block_is_an_error() {
    // Write a length of 10 but provide no body bytes.
    let tap = vec![0x0A, 0x00];
    assert!(parse_tap(&tap).is_err(), "truncated block should error");
}

#[test]
fn empty_tap_file_is_ok() {
    let blocks = parse_tap(&[]).expect("empty TAP is valid");
    assert!(blocks.is_empty());
}

#[test]
fn xor_checksum_correctness() {
    // Build a block where we can manually verify the checksum.
    // flag=0xFF, payload=[0xAA, 0x55]
    // checksum = 0xFF ^ 0xAA ^ 0x55 = 0x00
    let tap = make_tap_block(0xFF, &[0xAA, 0x55]);
    let expected_checksum = 0xFF ^ 0xAA ^ 0x55u8;
    assert_eq!(expected_checksum, 0x00);

    let blocks = parse_tap(&tap).expect("parse should succeed");
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].data, vec![0xAA, 0x55]);
}
