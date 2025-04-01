import pandas as pd
import matplotlib.pyplot as plt


def plot_charts(df, cols, xlabel, ylabel, title, filename, logy=False,):

    # Plot the bar chart
    df[cols].plot(kind='bar', logy=logy, legend=True, rot=0)

    # Show the plot
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(filename)


df = pd.read_csv('mem_time_profiling.csv')
df.set_index('workflow_name', inplace=True)

# compare times
time_cols = [col for col in df.columns if 'time' in col]
plot_charts(df, time_cols, 'Workflow', 'Elapsed Time (seconds) in log scale',
            'Pandas vs Dask Execution Time Comparison', 'comparison-time.png', True)

# performance
performance_cols = ['accuracy', 'f1_score', 'precision', 'recall', 'auc']
plot_charts(df, performance_cols, 'Workflow', 'Performance',
            'Pandas vs Dask Performance', 'comparison-performance.png')

# memory usage
memory_cols = ["memory_usage_during_execution (MB)"]
plot_charts(df, memory_cols, 'Workflow', 'Memory usage',
            'Pandas vs Dask Memory Usage', 'comparison-memory.png')
