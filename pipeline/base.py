class StreamStage:
    def reset(self):
        pass

    def process(self, data):
        pass

    def close(self):
        pass

class Pipeline:
    def __init__(self,stages=()):
        self.stages=list(stages)

    def process(self, data):
        res = data
        for s in self.stages:
            if(not res):
                break
            res = s.process(res)
        return res
    
    def reset(self):
        for s in self.stages:
            s.reset()

    def close(self):
        for s in reversed(self.stages):
            s.close()