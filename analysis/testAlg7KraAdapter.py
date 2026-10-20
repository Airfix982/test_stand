"""No original models needed. Tests the original RNG stream, gaps, rewinds, and ALG7 crypto method wiring."""
import numpy as np
from crypto.alg7_kra_adaptor import OriginalNumpyRngCursor, Alg7KraAdaptor

def test_original_stream():
    seed=37
    for bits in (1,7,27,81,129):
        ref=np.random.default_rng(seed)
        expected=[ref.integers(0,2,size=bits,dtype=np.uint8) for _ in range(120)]
        cur=OriginalNumpyRngCursor(seed,bits)
        for i in range(120):
            assert np.array_equal(cur.get_key_for_frame(i),expected[i]),(bits,i)
        assert np.array_equal(cur.get_key_for_frame(119),expected[119])
        assert np.array_equal(cur.get_key_for_frame(35),expected[35])
        assert np.array_equal(cur.get_key_for_frame(40),expected[40])
    print('PASS ALG7 original default_rng exact keys in order, repeat, backward replay')

def test_frame_gaps():
    bits=95
    ref=np.random.default_rng(4444)
    expected=[ref.integers(0,2,size=bits,dtype=np.uint8) for _ in range(650)]
    cur=OriginalNumpyRngCursor(4444,bits)
    for i in [0,1,2,3,5,8,24,25,62,63,64,65,80,88,155,320,319,321,500,649]:
        assert np.array_equal(cur.get_key_for_frame(i),expected[i]),i
    print('PASS ALG7 original keys after dropped frames, 64-boundary, and recovered index')

class FakeEncryption:
    encryption_bits=(4,9,20)
    def Encrypt_translation(self,x,bits): return (x+sum(map(int,bits)))%256
    def Decrypt_translation(self,x,bits): return (x-sum(map(int,bits)))%256
    def Encrypt_rotation(self,x,bits):return (np.asarray(x)+sum(map(int,bits)))%256
    def Decrypt_rotation(self,x,bits):return (np.asarray(x)-sum(map(int,bits)))%256

def test_encryption_wiring():
    tx=object.__new__(Alg7KraAdaptor)
    rx=object.__new__(Alg7KraAdaptor)
    for adaptor in (tx,rx):
        adaptor.bits_per_frame=20
        adaptor.crypt=FakeEncryption()
        adaptor._key_cursor=OriginalNumpyRngCursor(37,20)
    original=(42,91,np.array([1,2,3,4,5,6,7,8],dtype=float))
    for i in [0,1,2,90,97,98,170,169,171]:
        enc=tx._keyed(*original,i,False)
        dec=rx._keyed(*enc,i,True)
        assert dec[0]==original[0] and dec[1]==original[1] and np.allclose(dec[2],original[2]),i
    print('PASS ALG7 original Encrypt/Decrypt method calls, skip & rewind')

if __name__=='__main__':
    test_original_stream()
    test_frame_gaps()
    test_encryption_wiring()
    print('ALL ALG7 API TESTS PASSED')