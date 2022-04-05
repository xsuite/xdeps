class M:
    def __init__(self):
        self.values={}
        self.expr={}
    def __getitem__(self,k):
        ret=self.values.get(k,self.expr.get(k,0))
        if type(ret) is str:
            ret=eval(ret,{},self)
        return ret
    def __setitem__(self,k,v):
        if type(v) is str:
            self.expr[k]=v
        else:
            self.values[k]=v

if __name__=="__main__":
    m=M()
    m['c']="0.1*a+0.3*b"
    m['d']="0.2*a+0.4*b"

    print(m['c'])
    m['a']=3
    print(m['c'])





