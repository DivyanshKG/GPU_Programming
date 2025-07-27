import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv('results/results.csv')

# Compute speedup
results['Speedup'] = results['CPU_Time_s'] / results['CUDA_Time_s']

# Plot runtimes
plt.figure(figsize=(8, 5))
plt.plot(results['MatrixSize'], results['CPU_Time_s'], label='CPU', marker='o')
plt.plot(results['MatrixSize'], results['CUDA_Time_s'], label='CUDA (GPU)', marker='o')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Time (seconds)')
plt.title('Matrix Multiplication Runtime')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/runtime_plot.png')
plt.show()

# Plot speedup
plt.figure(figsize=(8, 5))
plt.plot(results['MatrixSize'], results['Speedup'], marker='o', color='green')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Speedup (CPU time / GPU time)')
plt.title('GPU Speedup over CPU')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/speedup_plot.png')
plt.show() 