import matplotlib.pyplot as plt

def plot(t, baseline='numpy'):
    """Compute speedup between two approaches."""
    # Prepare data
    base = t[baseline].average
    name = t.keys()
    time = [base / v.average for v in t.values()]
    # Plot speedups as bar graph
    plt.bar(name, time, width=0.8)
    plt.ylabel("Speedup")
    # Write speedup on top of bars
    for (i, n), t in zip(enumerate(name), time):
        plt.text(i, t, f"{t:.2f}x", horizontalalignment='center', fontweight='bold')
