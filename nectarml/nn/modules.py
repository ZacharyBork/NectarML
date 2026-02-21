
class Module():
    def __init__(self):
        pass
    
    def forward(self, *args, **kwargs): 
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)
