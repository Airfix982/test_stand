from algoss.alg7_kra.adapter import Cipher
from experiment.ALG7.ALG7_syncCipherServiceRuntime import main
from crypto.alg7_rx_enhance import OutputDynamics

if __name__ == '__main__':
    main(Cipher, 'alg7', 'call')


def configure_rx(self, repair=False, adaptive_lpc=False,
                 limiter=False, agc=False):
    self.impl.use_rx_repair = bool(repair)
    self.impl.use_adaptive_lpc = bool(adaptive_lpc)
    self.impl.use_output_limiter = bool(limiter or agc)

    self.impl.output_dynamics = OutputDynamics(
        peak_ceiling=0.93,
        target_rms=0.06 if agc else None
    )