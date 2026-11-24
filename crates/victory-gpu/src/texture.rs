//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    #[cfg(feature = "diagnostics")]
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    last: Mutex<Last>,
}

const NOT_WATCHED: u8 = 2;

/// What a texture's staging buffer holds: the content hash of the memory
/// detiled (memory not watched), or the write-watch asking it was detiled
/// after, with room to read the memory into.
#[derive(Default)]
struct Last {
    hash: Option<[u64; 2]>,
    asked: u64,
    /// `watch::remap_epoch()` when a `MEM_MAPPED` range was last verified
    /// committed (0 if not cached).
    committed_epoch: u64,
    read: Vec<u8>,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    set.get_or_insert_with(Default::default)
        .insert(resource as usize);
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

impl Staging {
    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        let epoch = crate::watch::remap_epoch();
        if last.committed_epoch != 0 && last.committed_epoch != epoch {
            last.committed_epoch = 0;
            self.watched.store(0, Ordering::Relaxed);
        }
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED
            && let Some((now, written)) = crate::watch::writes(start, start + size, false)
        {
            self.watched.store(1, Ordering::Relaxed);
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return 0;
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            last.read.resize(size, 0);
            crate::host::copy_mapped(last.read.as_mut_ptr(), self.memory, size, false);
            self.detile(&last.read);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        if last.committed_epoch != epoch {
            match committed_kind(self.memory, size) {
                Committed::No => {
                    #[cfg(feature = "diagnostics")]
                    tracing::debug!(
                        memory = ?self.memory,
                        size,
                        desc = ?self.layout.desc,
                        "a sampled texture's memory is not committed"
                    );
                    return 0;
                }
                Committed::Mapped => {
                    // Only physical-page views (MEM_MAPPED) are guaranteed to
                    // notify `title_memory_remapped` on unmap, so only cache
                    // their commit status across submissions.
                    last.committed_epoch = epoch;
                    self.watched.store(NOT_WATCHED, Ordering::Relaxed);
                }
                Committed::Other => {
                    self.watched.store(NOT_WATCHED, Ordering::Relaxed);
                }
            }
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = content_hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return size;
        }
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Committed {
    No,
    Mapped,
    Other,
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    committed_kind(at, len) != Committed::No
}

fn committed_kind(at: *const u8, len: usize) -> Committed {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let mut p = at as usize;
    let end = p + len;
    let mut all_mapped = true;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return Committed::No;
        }
        if info.Type != MEM_MAPPED {
            all_mapped = false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    if all_mapped {
        Committed::Mapped
    } else {
        Committed::Other
    }
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let s = Arc::new(Staging {
        buffer,
        #[cfg(feature = "diagnostics")]
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        watched: AtomicU8::new(0),
        last: Mutex::default(),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

/// Fast 128-bit non-cryptographic content digest for detecting changes in
/// unwatched texture memory. Processes 32 bytes per iteration across four
/// independent 64-bit accumulators to stay memory-bandwidth-bound.
fn content_hash(bytes: &[u8]) -> [u64; 2] {
    const K0: u64 = 0x9E37_79B1_85EB_CA87;
    const K1: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const K2: u64 = 0x1656_67B1_9E37_79F9;
    const K3: u64 = 0x85EB_CA77_C2B2_AE63;

    #[inline(always)]
    fn mix64(mut x: u64) -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    let mut a0 = K0 ^ (bytes.len() as u64);
    let mut a1 = K1;
    let mut a2 = K2;
    let mut a3 = K3;

    let (chunks, tail32) = bytes.as_chunks::<32>();
    for chunk in chunks {
        let w0 = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
        let w1 = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
        let w2 = u64::from_le_bytes(chunk[16..24].try_into().unwrap());
        let w3 = u64::from_le_bytes(chunk[24..32].try_into().unwrap());
        a0 = a0.wrapping_add(w0).wrapping_mul(K0).rotate_left(31);
        a1 = a1.wrapping_add(w1).wrapping_mul(K1).rotate_left(29);
        a2 = a2.wrapping_add(w2).wrapping_mul(K2).rotate_left(33);
        a3 = a3.wrapping_add(w3).wrapping_mul(K3).rotate_left(27);
    }

    let (words, tail) = tail32.as_chunks::<8>();
    for (i, word) in words.iter().enumerate() {
        let w = u64::from_le_bytes(*word);
        match i {
            0 => a0 = a0.wrapping_add(w).wrapping_mul(K0).rotate_left(31),
            1 => a1 = a1.wrapping_add(w).wrapping_mul(K1).rotate_left(29),
            _ => a2 = a2.wrapping_add(w).wrapping_mul(K2).rotate_left(33),
        }
    }

    if !tail.is_empty() {
        let mut buf = [0u8; 8];
        buf[..tail.len()].copy_from_slice(tail);
        let w = u64::from_le_bytes(buf);
        a3 = a3.wrapping_add(w).wrapping_mul(K3).rotate_left(27);
    }

    let lo = mix64(a0 ^ a2.rotate_left(17) ^ (bytes.len() as u64).wrapping_mul(K1));
    let hi = mix64(a1 ^ a3.rotate_left(23) ^ lo.wrapping_mul(K2));
    [lo, hi]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_hash_detects_single_bit_tail_and_order_changes() {
        let mut data: Vec<u8> = (0..4099).map(|i| (i * 37 + 13) as u8).collect();
        let base = content_hash(&data);
        assert_eq!(base, content_hash(&data));

        // Single-bit flips across main loop, 8-byte remainder, and 1..7-byte tail.
        for pos in [0, 7, 8, 31, 32, 1024, 4088, 4095, 4096, 4098] {
            data[pos] ^= 1;
            assert_ne!(content_hash(&data), base, "missed bit flip at {pos}");
            data[pos] ^= 1;
        }

        // Swapping two 8-byte lanes must change the digest.
        let mut swapped = data.clone();
        for i in 0..8 {
            swapped.swap(i, i + 8);
        }
        assert_ne!(content_hash(&swapped), base);

        // Appending a zero byte must change the digest.
        let mut extended = data.clone();
        extended.push(0);
        assert_ne!(content_hash(&extended), base);

        // Small sizes (0..=40) are all distinct when filled with zeros.
        let mut seen = std::collections::HashSet::new();
        for len in 0..=40 {
            let h = content_hash(&vec![0u8; len]);
            assert!(seen.insert(h), "collision on zero buffer of length {len}");
        }
    }

    #[test]
    fn committed_kind_distinguishes_mapped_and_private_and_remap_advances_epoch() {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile,
        };

        let private = vec![0u8; 4096];
        assert!(committed_kind(private.as_ptr(), private.len()) == Committed::Other);

        // SAFETY: pagefile-backed section of 1 page, unmapped and closed below.
        unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                4096,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 4096);
            assert!(!view.Value.is_null());
            assert!(committed_kind(view.Value.cast(), 4096) == Committed::Mapped);

            let before = crate::watch::remap_epoch();
            crate::watch::title_memory_remapped(view.Value as usize, 4096);
            assert!(crate::watch::remap_epoch() > before);

            UnmapViewOfFile(view);
            CloseHandle(section);
        }
    }

    #[test]
    #[ignore = "benchmark"]
    fn content_hash_vs_blake3_throughput() {
        let buf: Vec<u8> = (0..(16 << 20)).map(|i| (i * 131) as u8).collect();
        let t0 = std::time::Instant::now();
        let mut b3 = [0u8; 32];
        for _ in 0..10 {
            b3 = *std::hint::black_box(blake3::hash(std::hint::black_box(&buf))).as_bytes();
        }
        let blake3_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        let mut fast = [0u64; 2];
        for _ in 0..10 {
            fast = std::hint::black_box(content_hash(std::hint::black_box(&buf)));
        }
        let fast_ms = t1.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "160 MiB: blake3={blake3_ms:.2} ms ({:.2} GB/s), content_hash={fast_ms:.2} ms ({:.2} GB/s), speedup={:.1}x (b3[0]={}, fast={:?})",
            0.160 / (blake3_ms / 1000.0),
            0.160 / (fast_ms / 1000.0),
            blake3_ms / fast_ms,
            b3[0],
            fast,
        );
    }
}
