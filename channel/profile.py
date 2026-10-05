from dataclasses import dataclass

@dataclass(frozen=True)
class ChannelProfile:
    name:str
def getChannelProfile(name):
    return ChannelProfile(name)