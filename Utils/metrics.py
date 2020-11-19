import torch
import time
import matplotlib.pyplot as plt
import sys

def replace_line(print_string):
    sys.stdout.write("\r" + " "*40)
    sys.stdout.write("\r" + print_string)

def generate_metrics(n, dim, projectors, plot=False):
    """
    n: Tests per dimension
    dim: Maximum dimensionality to test
    projectors: dict with string keys and function values for which projectors to test
    """

    dims = torch.arange(1, dim+1)

    results = {}

    for name in list(projectors):
        torch.manual_seed(0) # Generate same samples for all projectors
        
        times = []
        for dim in dims:
            replace_line("Running " + str(name) + " DIM: " + str(dim))
            start = time.time_ns()
            x = torch.rand(n, dim)*4 - 2
            x_proj = projectors[name](x)
            end = time.time_ns()

            duration = end - start
            duration_avg = duration / n
            duration_ms = duration_avg / 1e3

            times.append(duration_ms)

        p_result = {}
        p_result["runtime"] = times
        p_result["dims"] = dims

        results[name] = p_result
        
    if plot:
        plt.figure()
        
        for name in list(results):
            p_result = results[name]
            plt.plot(p_result["dims"].numpy(), p_result["runtime"], label=name)

        plt.legend()
        plt.xlabel("Vector Dimensionality")
        plt.ylabel("Microseconds ellapsed per projection (avg)")

    return results