import os

dims = [150]
section = "COPA"

for dim in dims:
    dim = str(dim)
    os.system("python3 thread_run.py  " + section + " " + dim +"> newlog"+str(dim)+" &")
