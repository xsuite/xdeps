import time
import json
import xdeps.madxutils


st0=time.time()
data=json.load(open('data.json'))
st1=time.time()
print(f"Loading json {st1-st0} sec")

m=xdeps.madxutils.MadxEnv()
st0=time.time()
m.load(data)
st1=time.time()
print(f"Loading json {st1-st0} sec")
