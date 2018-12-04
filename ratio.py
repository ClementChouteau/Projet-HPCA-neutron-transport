import subprocess
import re
import sys

from subprocess import DEVNULL

temps_re = re.compile("Millions de neutrons \/s: ([0-9.]+)")

Million_per_sec = {}
for version in ["./bin/neutron-omp", "./bin/neutron-gpu"]:
	h = sys.argv[1]
	n = sys.argv[2]
	c_c = sys.argv[3]
	c_s = sys.argv[4]
	threadsPerBlock = sys.argv[5]
	neutronsPerThread = sys.argv[6]
	
	p = subprocess.Popen(['optirun', version, h, n, c_c, c_s, threadsPerBlock, neutronsPerThread], stdout=subprocess.PIPE, stderr=DEVNULL)
	out = p.communicate()[0].decode("utf-8")
	M = float(temps_re.search(out).group(1))
	Million_per_sec[version] = M

v_omp = Million_per_sec["./bin/neutron-omp"]
v_gpu = Million_per_sec["./bin/neutron-gpu"]
print(v_omp/(v_omp+v_gpu))
