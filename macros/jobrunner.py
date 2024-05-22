import concurrent.futures
import numpy as np
import subprocess
import os
import tqdm

def run_process(args):
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'photonbomb_test.csv')
    out = subprocess.run(['python', 'nphoton_scan.py', '-o', path] + args,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=True)
    return out.stdout

arguments = [f'-n {i}'.split() for i in np.logspace(0, 7, 40, dtype=int)]

if __name__ == "__main__":
    from dask.distributed import Client, as_completed
    print('Saving to data/photonbomb_test.csv')
    import dask
    dask.config.set(scheduler='processes')

    client = Client(n_workers=4)

    futures = client.map(run_process, arguments)
    
    with tqdm.tqdm(total=len(futures)) as pbar:
        for future in as_completed(futures):
            stdout = future.result().decode('utf-8').splitlines()[-1]
            n, detected, pte, t = stdout.split('\n')[-1].split()
            n = float(n)
            detected = float(detected)
            pte = float(pte)
            t = float(t)
            pbar.write(f"N {n:<6.0f} D {detected:<6.0f} PTE {pte:.2f} t {t:.2f}")
            pbar.update(1)

    client.close()
    print("Done!")