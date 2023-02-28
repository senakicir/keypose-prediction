import sys,torch

def raiseError(message="Runtime error"):
    if(None==sys.exc_info()[0]):
        print(message, )
    else: 
        print(message,sys.exc_info()[0])
    raise RuntimeError


def makeCudaIfPossible(x):
    
    if(x is None):
        return None
    elif(cudaOnP()):
        if(x.is_cuda):
            return x
        else:
            return x.cuda()
    else:
        return x
    
def cudaOnP():
    return (torch.cuda.is_available())