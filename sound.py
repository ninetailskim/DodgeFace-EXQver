import _thread
from playsound import playsound

def thread_playsound(threadName, soundname):
    print(threadName,soundname)
    playsound("resources/music/"+soundname)