//! POK file parser — ZX Spectrum cheat/poke format.
//!
//! Format per line:
//!   Y <description>   — display label, ignored
//!   M <bank> <addr> <value> <old|N>  — memory poke
//!   Z                 — end of poke set

#[derive(Debug, PartialEq)]
pub struct Poke {
    pub addr: u16,
    pub value: u8,
}

/// Parse a `.pok` file into a list of `(addr, value)` pokes.
///
/// Only bank 8 (48K RAM) is supported. Lines with other banks are skipped.
/// Malformed lines are skipped silently.
pub fn parse_pok(data: &str) -> Result<Vec<Poke>, String> {
    let mut pokes = Vec::new();

    for line in data.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('Y') || line.starts_with('Z') {
            continue;
        }
        if !line.starts_with('M') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            continue;
        }
        let bank: u8 = match parts[1].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if bank != 8 {
            continue; // only 48K RAM supported
        }
        let addr: u32 = match parts[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if addr > 65535 {
            continue;
        }
        let value: u8 = match parts[3].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        pokes.push(Poke { addr: addr as u16, value });
    }

    Ok(pokes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const J83_POK: &str = "\
Y Bypass startup protection\n\
M 8 23672 90 N\n\
M 8 23673 131 N\n\
M 8 23728 233 N\n\
Z\n";

    #[test]
    fn test_valid_three_pokes() {
        let pokes = parse_pok(J83_POK).unwrap();
        assert_eq!(pokes.len(), 3);
        assert_eq!(pokes[0], Poke { addr: 23672, value: 90 });
        assert_eq!(pokes[1], Poke { addr: 23673, value: 131 });
        assert_eq!(pokes[2], Poke { addr: 23728, value: 233 });
    }

    #[test]
    fn test_y_and_z_lines_ignored() {
        let data = "Y some description\nZ\n";
        let pokes = parse_pok(data).unwrap();
        assert_eq!(pokes.len(), 0);
    }

    #[test]
    fn test_empty_file() {
        let pokes = parse_pok("").unwrap();
        assert_eq!(pokes.len(), 0);
    }

    #[test]
    fn test_non_bank_8_skipped() {
        let data = "M 0 23672 90 N\nM 8 23673 131 N\n";
        let pokes = parse_pok(data).unwrap();
        assert_eq!(pokes.len(), 1);
        assert_eq!(pokes[0].addr, 23673);
    }

    #[test]
    fn test_malformed_line_skipped() {
        let data = "M 8 notanumber 90 N\nM 8 23672 90 N\n";
        let pokes = parse_pok(data).unwrap();
        assert_eq!(pokes.len(), 1);
        assert_eq!(pokes[0].addr, 23672);
    }

    #[test]
    fn test_addr_too_large_skipped() {
        let data = "M 8 99999 90 N\nM 8 23672 90 N\n";
        let pokes = parse_pok(data).unwrap();
        assert_eq!(pokes.len(), 1);
    }
}
