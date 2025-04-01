
_dir="$(pwd)"
echo $_dir
export PYTHONPATH=$_dir
# Run Pandas Workflow
python pandas_workflow/scripts/pandas_main.py --file 'config.yaml'
# Run Dask Workflow
python dask_workflow/scripts/dask_main.py --file 'config.yaml'

# run comparison analysis
cd analysis
python comparison.py