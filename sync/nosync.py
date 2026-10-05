from pipeline.base import StreamStage

class Nosync(StreamStage):
    name="sync:no"

    def __init__(self, direction):
        self.direction = direction

    def process(self, data):
        return data