import time
import xdeps.madxutils

st=time.time()
m2=xdeps.madxutils.MadxEnv.from_json('data.json')
print(time.time()-st)

