"""
Parallel JMD batch decomposition, 3 workers.
Reads progress from decomposition_progress.csv, skips completed files,
saves .npz results to data/decomposed/absent|present|unknown/.
Safe to stop and restart at any time, progress is saved after every file.

Usage, run in terminal from project root:
    python decompose_parallel.py or
    python decompose_parallel.py --workers 4
"""

import numpy as np
import pandas as pd
import json
import sys
import argparse
from pathlib import Path
from time import time
from datetime import datetime
from multiprocessing import Pool

PROJECT_ROOT   = Path(r"D:\sop")
PROCESSED_DIR  = PROJECT_ROOT / 'data' / 'processed'
DECOMPOSED_DIR = PROJECT_ROOT / 'data' / 'decomposed'
PROGRESS_FILE  = PROCESSED_DIR / 'decomposition_progress.csv'

sys.path.insert(0, str(PROJECT_ROOT / 'decomposition'))


def process_file(args):
    """Worker function, runs JMD on one file and saves the result."""
    idx, row, jmd_params = args

    from jmd import JMD

    result = {
        'idx':          idx,
        'file':         row['file'],
        'status':       None,
        'duration_sec': None,
        'error':        None,
        'timestamp':    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    t0 = time()
    try:
        signal = np.load(row['npy_path']).astype(np.float64)
        u, v, omega = JMD(signal, **jmd_params)
        elapsed = round(time() - t0, 1)

        np.savez_compressed(row['output_path'], u=u, v=v, omega=omega)

        result['status']       = 'completed'
        result['duration_sec'] = elapsed

        freqs = [f"{w * 1000:.0f}" for w in omega]
        print(f"  done  {row['file']}  {elapsed:.1f}s  {freqs} Hz", flush=True)

    except Exception as e:
        result['status']       = 'failed'
        result['duration_sec'] = round(time() - t0, 1)
        result['error']        = str(e)
        print(f"  FAIL  {row['file']}  {e}", flush=True)

    return result


def save_result(progress, result):
    """Updates one row in the progress CSV."""
    idx = result['idx']
    progress.loc[idx, 'status']       = result['status']
    progress.loc[idx, 'duration_sec'] = result['duration_sec']
    progress.loc[idx, 'timestamp']    = result['timestamp']
    progress.loc[idx, 'error']        = result['error']
    progress.to_csv(PROGRESS_FILE, index=False)


def main(n_workers):
    print(f"Parallel JMD decomposition: {n_workers} workers")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Progress file: {PROGRESS_FILE}")
    print()

    with open(PROCESSED_DIR / 'jmd_params.json') as f:
        jmd_params = json.load(f)
    print(f"JMD params: {jmd_params}")

    progress  = pd.read_csv(PROGRESS_FILE)
    pending   = progress[progress['status'] == 'pending'].copy()

    n_total     = len(progress)
    n_completed = (progress['status'] == 'completed').sum()
    n_pending   = len(pending)

    print(f"Total files  : {n_total}")
    print(f"Already done : {n_completed}")
    print(f"To process   : {n_pending}")
    print()

    if n_pending == 0:
        print("Nothing to do; all files already completed.")
        return

    est_hours = n_pending * 100 / n_workers / 3600
    print(f"Estimated time: ~{est_hours:.1f} hours at ~100s/file with {n_workers} workers")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    args_list = [
        (idx, row, jmd_params)
        for idx, row in pending.iterrows()
    ]

    n_success = 0
    n_fail    = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(process_file, args_list):
            if result['status'] == 'completed':
                n_success += 1
            else:
                n_fail += 1

            # reload latest CSV before writing to avoid overwriting concurrent saves
            progress = pd.read_csv(PROGRESS_FILE)
            save_result(progress, result)

            done = n_success + n_fail
            print(f"  [{done}/{n_pending}] completed={n_success} failed={n_fail}", flush=True)

    print()
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Succeeded : {n_success}")
    print(f"  Failed    : {n_fail}")

    progress = pd.read_csv(PROGRESS_FILE)
    print(f"  Total completed: {(progress['status'] == 'completed').sum()}/{len(progress)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3,
                        help='Number of parallel workers (default: 3)')
    args = parser.parse_args()
    main(args.workers)
