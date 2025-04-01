from dask.distributed import Client
def setup_dask_client(n_workers=4, threads_per_worker=2, memory_limit='8GB'):
    client = Client(n_workers=n_workers, 
                   threads_per_worker=threads_per_worker,
                   memory_limit=memory_limit)
    return client