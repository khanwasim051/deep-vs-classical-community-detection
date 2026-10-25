"""Print the hardware and software environment for the manuscript (Section 4.7).

    python env_report.py > env_report.txt
"""
import importlib, os, platform, subprocess, sys

def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as ex:
        return f"(failed: {ex})"

print("OS:", platform.platform())
print("Python:", sys.version.replace("\n", " "))
cpu = platform.processor()
if os.name == "nt":
    cpu = run('powershell -command "(Get-CimInstance Win32_Processor).Name"') or cpu
    ram = run('powershell -command "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"')
    cores = run('powershell -command "(Get-CimInstance Win32_Processor).NumberOfCores; (Get-CimInstance Win32_Processor).NumberOfLogicalProcessors"')
    print("CPU:", cpu); print("RAM (GB):", ram); print("Cores / logical processors:", cores.replace("\n", " / "))
else:
    print("CPU:", run("lscpu | grep 'Model name'") or cpu)
try:
    import torch
    print("PyTorch:", torch.__version__, "| CUDA (build):", torch.version.cuda, "| cuDNN:", torch.backends.cudnn.version())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("GPU:", p.name, f"| memory {p.total_memory / 2**30:.1f} GB")
except Exception as ex:
    print("PyTorch: not importable", ex)
print("NVIDIA driver:", run("nvidia-smi --query-gpu=driver_version --format=csv,noheader"))
for mod in ["torch_geometric", "pyagc", "numpy", "scipy", "sklearn", "networkx", "igraph", "leidenalg", "pandas", "matplotlib"]:
    try:
        m = importlib.import_module(mod)
        print(f"{mod}:", getattr(m, "__version__", "?"))
    except Exception as ex:
        print(f"{mod}: not installed ({type(ex).__name__})")
