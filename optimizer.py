import subprocess
import re
import random
import math
import sys
from threading import Timer

from subprocess import DEVNULL

def gen_random_exp(l, h, n):
	# y = a exp(b X)
	a = l
	b = math.log(h/l)
	for _ in range(n):
		X = random.random()
		yield int(a * math.exp(b*X))

h = sys.argv[1]
n = sys.argv[2]
c_c = sys.argv[3]
c_s = sys.argv[4]

temps_re = re.compile('Temps total de calcul: ([0-9.]+)')

best = (float('inf'), 32, 1000)
for threadsPerBlock in range(32, 192, 32):
	for neutronsPerThread in gen_random_exp(1000, int(n), 10):
		try:
			proc = subprocess.Popen(['optirun', './bin/neutron-gpu', h, n, c_c, c_s, str(threadsPerBlock), str(neutronsPerThread)], stdout=subprocess.PIPE, stderr=DEVNULL)
			timer = Timer(min(best[0], 100000), proc.kill)
			timer.start()
			out = proc.communicate()[0].decode("utf-8")
			t = float(temps_re.search(out).group(1))
			(best_t, tpb, npt) = best
			if (t < best_t):
				best = (t, threadsPerBlock, neutronsPerThread)
			(best_t, tpb, npt) = best
			print('cur:', '{:8f}'.format(t), '{:3d}'.format(threadsPerBlock), '{:12d}'.format(neutronsPerThread), '   ', 'best:', '{:8f}'.format(best_t), '{:3d}'.format(tpb), '{:12d}'.format(npt))
		except KeyboardInterrupt:
			print("\n", best)
			sys.exit(0)
		except:
			pass
		finally:
			try:
				timer.cancel()
			except:
				pass

print(best)
