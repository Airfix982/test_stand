"""Non-hardware regressions for the spectral real-time bridge."""
import io
from unittest.mock import patch
import numpy as np
from experiment import spectralTimepack20k as live

class Cipher:
    def encode_batch(self, x, indices):
        return x.copy()

class Sink:
    def __init__(self): self.n=0
    def writeframes(self, b): self.n+=len(b)

bank=np.ones((64,80),dtype=np.float32)*0.1

def batches(n):
    for j in range(n):
        yield np.ones((4,320),dtype=np.float32)*.01

out=Sink()
s=live._generate_frames(Cipher(),batches(3),bank,outfile=out,duration=0)
assert s['tx_frames']==12 and out.n==12*400*2, s
print('PASS continuous live TX: 0 = unlimited until source ends')

out=Sink()
s=live._generate_frames(Cipher(),batches(3),bank,outfile=out,duration=.08)
assert s['tx_frames']==4 and out.n==4*400*2,s
print('PASS positive duration limits live TX')

class InterruptIterator:
    def __iter__(self):
        yield np.ones((4,320),dtype=np.float32)*.01
        raise KeyboardInterrupt
out=Sink()
s=live._generate_frames(Cipher(),iter(InterruptIterator()),bank,outfile=out,duration=0)
assert s['tx_frames']==4 and s['stopped_by_user'] and out.n==4*400*2,s
print('PASS Ctrl+C returns stats instead of aborting virtual device cleanup')

info='Default Source: tx_mic\nDefault Sink: rx_sink\n'
sources='1\talsa_input.pci-00.analog-stereo\tmod\n2\ttx_mic\tmod\n3\trx_sink.monitor\tmod\n'
sinks='1\talsa_output.pci-00.analog-stereo\tmod\n2\trx_sink\tmod\n3\ttx_enc_sink\tmod\n'
def fake_pactl(*a):
    if a==('info',):return info
    if a==('list','short','sources'):return sources
    if a==('list','short','sinks'):return sinks
    raise AssertionError(a)
with patch.object(live,'pactl',fake_pactl):
    assert live._physical_endpoint('sources')=='alsa_input.pci-00.analog-stereo'
    assert live._physical_endpoint('sinks')=='alsa_output.pci-00.analog-stereo'
    try:live._physical_endpoint('sources','tx_mic')
    except RuntimeError:pass
    else:raise AssertionError('Virtual mic accepted as capture source')
print('PASS physical audio devices selected even after defaults become virtual')
print('ALL SPECTRAL LIVE TESTS PASSED')
