use crate::frame::groupby::populate_multiple_key_hashmap;
use crate::frame::hash_join::{get_hash_tbl_threaded_join, n_join_threads};
use crate::prelude::*;
use crate::utils::split_df;
use crate::vector_hasher::{df_rows_to_hashes_threaded, this_thread, IdBuildHasher, IdxHash};
use crate::POOL;
use hashbrown::HashMap;
use rayon::prelude::*;

/// Compare the rows of two DataFrames
unsafe fn compare_df_rows2(
    left: &DataFrame,
    right: &DataFrame,
    left_idx: usize,
    right_idx: usize,
) -> bool {
    for (l, r) in left.get_columns().iter().zip(right.get_columns()) {
        if !(l.get_unchecked(left_idx) == r.get_unchecked(right_idx)) {
            return false;
        }
    }
    true
}

fn create_build_table(
    hashes: &[UInt64Chunked],
    keys: &DataFrame,
) -> Vec<HashMap<IdxHash, Vec<u32>, IdBuildHasher>> {
    let n_threads = hashes.len();
    let size = hashes.iter().fold(0, |acc, v| acc + v.len());

    // We will create a hashtable in every thread.
    // We use the hash to partition the keys to the matching hashtable.
    // Every thread traverses all keys/hashes and ignores the ones that doesn't fall in that partition.
    POOL.install(|| {
        (0..n_threads).into_par_iter().map(|thread_no| {
            let thread_no = thread_no as u64;
            // TODO:: benchmark size
            let mut hash_tbl: HashMap<IdxHash, Vec<u32>, IdBuildHasher> =
                HashMap::with_capacity_and_hasher(size / (5 * n_threads), IdBuildHasher::default());

            let n_threads = n_threads as u64;
            let mut offset = 0;
            for hashes in hashes {
                for hashes in hashes.data_views() {
                    let len = hashes.len();
                    let mut idx = 0;
                    hashes.iter().for_each(|h| {
                        // partition hashes by thread no.
                        // So only a part of the hashes go to this hashmap
                        if this_thread(*h, thread_no, n_threads) {
                            let idx = idx + offset;
                            populate_multiple_key_hashmap(
                                &mut hash_tbl,
                                idx,
                                *h,
                                keys,
                                || vec![idx],
                                |v| v.push(idx),
                            )
                        }
                        idx += 1;
                    });

                    offset += len as u32;
                }
            }
            hash_tbl
        })
    })
    .collect()
}

/// Probe the build table and add tuples to the results (inner join)
#[allow(clippy::too_many_arguments)]
fn probe_inner<F>(
    probe_hashes: &UInt64Chunked,
    hash_tbls: &[HashMap<IdxHash, Vec<u32>, IdBuildHasher>],
    results: &mut Vec<(u32, u32)>,
    local_offset: usize,
    n_tables: u64,
    a: &DataFrame,
    b: &DataFrame,
    swap_fn: F,
) where
    F: Fn(u32, u32) -> (u32, u32),
{
    let mut idx_a = local_offset as u32;
    for probe_hashes in probe_hashes.data_views() {
        for &h in probe_hashes {
            // probe table that contains the hashed value
            let current_probe_table = unsafe { get_hash_tbl_threaded_join(h, hash_tbls, n_tables) };

            let entry = current_probe_table.raw_entry().from_hash(h, |idx_hash| {
                let idx_b = idx_hash.idx;
                // Safety:
                // indices in a join operation are always in bounds.
                unsafe { compare_df_rows2(a, b, idx_a as usize, idx_b as usize) }
            });

            if let Some((_, indexes_b)) = entry {
                let tuples = indexes_b.iter().map(|&idx_b| swap_fn(idx_a, idx_b));
                results.extend(tuples);
            }
            idx_a += 1;
        }
    }
}

fn get_offsets(probe_hashes: &[UInt64Chunked]) -> Vec<usize> {
    probe_hashes
        .iter()
        .map(|ph| ph.len())
        .scan(0, |state, val| {
            let out = *state;
            *state += val;
            Some(out)
        })
        .collect()
}

pub(crate) fn inner_join_multiple_keys(
    a: &DataFrame,
    b: &DataFrame,
    swap: bool,
) -> Vec<(u32, u32)> {
    // we assume that the b DataFrame is the shorter relation.
    // b will be used for the build phase.

    let n_threads = n_join_threads();
    let dfs_a = split_df(&a, n_threads).unwrap();
    let dfs_b = split_df(&b, n_threads).unwrap();

    let (build_hashes, random_state) = df_rows_to_hashes_threaded(&dfs_b, None);
    let (probe_hashes, _) = df_rows_to_hashes_threaded(&dfs_a, Some(random_state));

    let hash_tbls = create_build_table(&build_hashes, b);
    // early drop to reduce memory pressure
    drop(build_hashes);

    let n_tables = hash_tbls.len() as u64;
    let offsets = get_offsets(&probe_hashes);
    // next we probe the other relation
    // code duplication is because we want to only do the swap check once
    POOL.install(|| {
        probe_hashes
            .into_par_iter()
            .zip(offsets)
            .map(|(probe_hashes, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;
                let mut results =
                    Vec::with_capacity(probe_hashes.len() / POOL.current_num_threads());
                let local_offset = offset;
                // code duplication is to hoist swap out of the inner loop.
                if swap {
                    probe_inner(
                        &probe_hashes,
                        hash_tbls,
                        &mut results,
                        local_offset,
                        n_tables,
                        a,
                        b,
                        |idx_a, idx_b| (idx_b, idx_a),
                    )
                } else {
                    probe_inner(
                        &probe_hashes,
                        hash_tbls,
                        &mut results,
                        local_offset,
                        n_tables,
                        a,
                        b,
                        |idx_a, idx_b| (idx_a, idx_b),
                    )
                }

                results
            })
            .flatten()
            .collect()
    })
}
pub(crate) fn left_join_multiple_keys(a: &DataFrame, b: &DataFrame) -> Vec<(u32, Option<u32>)> {
    // we assume that the b DataFrame is the shorter relation.
    // b will be used for the build phase.

    let n_threads = n_join_threads();
    let dfs_a = split_df(&a, n_threads).unwrap();
    let dfs_b = split_df(&b, n_threads).unwrap();

    let (build_hashes, random_state) = df_rows_to_hashes_threaded(&dfs_b, None);
    let (probe_hashes, _) = df_rows_to_hashes_threaded(&dfs_a, Some(random_state));

    let hash_tbls = create_build_table(&build_hashes, b);
    // early drop to reduce memory pressure
    drop(build_hashes);

    let n_tables = hash_tbls.len() as u64;
    let offsets = get_offsets(&probe_hashes);

    // next we probe the other relation
    // code duplication is because we want to only do the swap check once
    POOL.install(|| {
        probe_hashes
            .into_par_iter()
            .zip(offsets)
            .map(|(probe_hashes, offset)| {
                // local reference
                let hash_tbls = &hash_tbls;
                let mut results =
                    Vec::with_capacity(probe_hashes.len() / POOL.current_num_threads());
                let local_offset = offset;

                let mut idx_a = local_offset as u32;
                for probe_hashes in probe_hashes.data_views() {
                    for &h in probe_hashes {
                        // probe table that contains the hashed value
                        let current_probe_table =
                            unsafe { get_hash_tbl_threaded_join(h, hash_tbls, n_tables) };

                        let entry = current_probe_table.raw_entry().from_hash(h, |idx_hash| {
                            let idx_b = idx_hash.idx;
                            // Safety:
                            // indices in a join operation are always in bounds.
                            unsafe { compare_df_rows2(a, b, idx_a as usize, idx_b as usize) }
                        });

                        match entry {
                            // left and right matches
                            Some((_, indexes_b)) => {
                                results.extend(indexes_b.iter().map(|&idx_b| (idx_a, Some(idx_b))))
                            }
                            // only left values, right = null
                            None => results.push((idx_a, None)),
                        }
                        idx_a += 1;
                    }
                }

                results
            })
            .flatten()
            .collect()
    })
}
