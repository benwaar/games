/// TAP file parser for ZX Spectrum tape images.
///
/// ## Format
/// Each TAP block:
///   - 2-byte LE length prefix (`block_len`)
///   - `block_len` bytes: `[flag, payload..., xor_checksum]`
///
/// Flag 0x00 = header block (19 bytes total: flag + 17-byte payload + checksum).
/// Flag 0xFF = data block.
///
/// XOR checksum = XOR of flag + all payload bytes (i.e., everything except the
/// stored checksum itself).
///
/// ## Header payload (17 bytes)
///   offset  0      : block type (0=Program, 1=Number array, 2=Char array, 3=Code)
///   offset  1–10   : filename (10 bytes, space-padded)
///   offset 11–12   : data_len (LE u16)
///   offset 13–14   : param1   (LE u16) — autostart line (Program) / load addr (Code)
///   offset 15–16   : param2   (LE u16) — variable offset (Program) / length (Code)

/// Parsed TAP block — flag, payload, and optional header metadata.
#[derive(Debug, Clone)]
pub struct TapBlock {
    /// 0x00 = header, 0xFF = data
    pub flag: u8,
    /// Payload bytes (flag and checksum stripped).
    pub data: Vec<u8>,
    /// Populated for header blocks (flag == 0x00).
    pub header: Option<TapHeader>,
}

/// Decoded header block fields.
#[derive(Debug, Clone)]
pub struct TapHeader {
    /// 0=Program, 1=Number array, 2=Char array, 3=Code
    pub block_type: u8,
    /// 10-byte name (space-padded)
    pub name: [u8; 10],
    /// Declared length of the following data block
    pub data_len: u16,
    /// param1: autostart line (Program) or load address (Code)
    pub param1: u16,
    /// param2: variable offset (Program) or length (Code)
    pub param2: u16,
}

/// Parse a TAP file into a Vec of blocks.
///
/// Returns `Err` if the data is truncated, a checksum mismatches,
/// or a header block has an unexpected payload length.
pub fn parse_tap(data: &[u8]) -> Result<Vec<TapBlock>, String> {
    let mut blocks = Vec::new();
    let mut pos = 0usize;

    while pos < data.len() {
        // Need at least 2 bytes for the length field.
        if pos + 2 > data.len() {
            return Err(format!(
                "truncated TAP: only {} byte(s) left at offset {}, need 2 for length",
                data.len() - pos,
                pos
            ));
        }

        let block_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;

        if block_len == 0 {
            return Err(format!("zero-length block at offset {}", pos - 2));
        }

        if pos + block_len > data.len() {
            return Err(format!(
                "truncated block at offset {}: declared {} bytes, only {} remain",
                pos - 2,
                block_len,
                data.len() - pos
            ));
        }

        let block_data = &data[pos..pos + block_len];
        pos += block_len;

        // Minimum valid block: flag + checksum = 2 bytes.
        if block_len < 2 {
            return Err(format!(
                "block too short: {} byte(s), minimum is 2",
                block_len
            ));
        }

        let flag = block_data[0];
        let payload = &block_data[1..block_len - 1];
        let stored_checksum = block_data[block_len - 1];

        // XOR of flag + payload (everything before the checksum byte).
        let computed = block_data[..block_len - 1]
            .iter()
            .fold(0u8, |acc, &b| acc ^ b);

        if computed != stored_checksum {
            return Err(format!(
                "checksum mismatch in block {}: stored 0x{:02X}, computed 0x{:02X}",
                blocks.len() + 1,
                stored_checksum,
                computed
            ));
        }

        // Parse header if this is a header block.
        let header = if flag == 0x00 {
            if payload.len() != 17 {
                return Err(format!(
                    "header block payload is {} bytes, expected 17",
                    payload.len()
                ));
            }
            let mut name = [0u8; 10];
            name.copy_from_slice(&payload[1..11]);
            Some(TapHeader {
                block_type: payload[0],
                name,
                data_len: u16::from_le_bytes([payload[11], payload[12]]),
                param1: u16::from_le_bytes([payload[13], payload[14]]),
                param2: u16::from_le_bytes([payload[15], payload[16]]),
            })
        } else {
            None
        };

        blocks.push(TapBlock {
            flag,
            data: payload.to_vec(),
            header,
        });
    }

    Ok(blocks)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a valid TAP block from flag + payload.
    fn make_block(flag: u8, payload: &[u8]) -> Vec<u8> {
        let body_len = 1 + payload.len() + 1; // flag + payload + checksum
        let mut block = Vec::with_capacity(2 + body_len);

        // Length prefix (LE u16)
        block.push((body_len & 0xFF) as u8);
        block.push((body_len >> 8) as u8);

        // flag
        block.push(flag);
        block.extend_from_slice(payload);

        // XOR checksum
        let checksum = std::iter::once(flag)
            .chain(payload.iter().copied())
            .fold(0u8, |acc, b| acc ^ b);
        block.push(checksum);

        block
    }

    fn make_header_payload(
        block_type: u8,
        name: &[u8; 10],
        data_len: u16,
        param1: u16,
        param2: u16,
    ) -> Vec<u8> {
        let mut p = vec![block_type];
        p.extend_from_slice(name);
        p.push((data_len & 0xFF) as u8);
        p.push((data_len >> 8) as u8);
        p.push((param1 & 0xFF) as u8);
        p.push((param1 >> 8) as u8);
        p.push((param2 & 0xFF) as u8);
        p.push((param2 >> 8) as u8);
        p
    }

    #[test]
    fn parse_two_block_tap() {
        let name = *b"TestProg  "; // 10 bytes
        let header_payload = make_header_payload(0, &name, 10, 0x8000, 0);
        let data_payload: Vec<u8> = (0u8..10).collect();

        let mut tap = make_block(0x00, &header_payload);
        tap.extend(make_block(0xFF, &data_payload));

        let blocks = parse_tap(&tap).expect("parse should succeed");
        assert_eq!(blocks.len(), 2);

        // Header block
        assert_eq!(blocks[0].flag, 0x00);
        assert_eq!(blocks[0].data, header_payload);
        let hdr = blocks[0].header.as_ref().unwrap();
        assert_eq!(hdr.block_type, 0);
        assert_eq!(&hdr.name, b"TestProg  ");
        assert_eq!(hdr.data_len, 10);
        assert_eq!(hdr.param1, 0x8000);

        // Data block
        assert_eq!(blocks[1].flag, 0xFF);
        assert_eq!(blocks[1].data, data_payload);
        assert!(blocks[1].header.is_none());
    }

    #[test]
    fn checksum_mismatch_returns_error() {
        let mut tap = make_block(0xFF, &[1, 2, 3]);
        // Corrupt the checksum (last byte of the block).
        *tap.last_mut().unwrap() ^= 0xFF;

        assert!(parse_tap(&tap).is_err());
    }

    #[test]
    fn truncated_tap_returns_error() {
        // Only the length prefix, no body.
        let tap = vec![0x05, 0x00];
        assert!(parse_tap(&tap).is_err());
    }

    #[test]
    fn empty_tap_returns_empty_vec() {
        let blocks = parse_tap(&[]).expect("empty tap is valid");
        assert!(blocks.is_empty());
    }
}
