from pipeline.base import StreamStage

class IdentityAlg(StreamStage):
    name="alg:no"

    def __init__(self, direction):
        self.direction = direction

    def process(self, data):
        return data

