//! The Walsh-Hadamard row transform and its side-length helper, ported
//! from gguf-recast (butterfly order identical to the Python reference).

/// In-place normalized fast Walsh-Hadamard transform of one row.
pub fn walsh_hadamard_row(row: &mut [f32]) {
    let d = row.len();
    assert!(d.is_power_of_two(), "WHT requires power-of-2 length");
    let mut h = 1;
    while h < d {
        let mut i = 0;
        while i < d {
            for j in i..i + h {
                let a = row[j];
                let b = row[j + h];
                row[j] = a + b;
                row[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    let norm = (d as f64).sqrt() as f32;
    for x in row.iter_mut() {
        *x /= norm;
    }
}

/// The largest power of two not exceeding `n` (the T5 side length).
pub fn largest_pow2_le(n: usize) -> usize {
    let mut d = 1;
    while d * 2 <= n {
        d *= 2;
    }
    d
}
