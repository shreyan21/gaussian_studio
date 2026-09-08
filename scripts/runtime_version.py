"""PowerShell-safe runtime probe (avoid native shell quoting of Python -c strings)."""
try:
    import torch
    print(torch.__version__)
except ImportError:
    print("missing")
except Exception:
    print("broken")
