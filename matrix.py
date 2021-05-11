import numpy as np

def dot(a,b):
    if len(a)!=len(b): raise NameError('dimension mismatch')
    s=0
    for i in range(0,len(a)):
        s+=a[i]*b[i]
    return s

class mtx:
    def __init__(self,m):
        self.m=m
        self.c=len(m)
        self.r=len(m[0])
    #the following four functions implement +,-,*,/ between two matrices
    #or between a matrix and a scalar
    #matrix addition and subtraction is element wise  O(n**2) for both
    def __add__(self,a):
        if(self.c!=a.c or self.r!=a.r): raise NameError('dimension mismatch')
        s=[];
        for i in range(0,self.c):
            si=[]
            for j in range(0,self.r):
                si+=[self.m[i][j]+a.m[i][j]]
            s+=[si]
        return mtx(s)
    def __sub__(self,a):
        if(self.c!=a.c or self.r!=a.r): raise NameError('dimension mismatch')
        s=[];
        for i in range(0,self.c):
            si=[]
            for j in range(0,self.r):
                si+=[self.m[i][j]-a.m[i][j]]
            s+=[si]
        return mtx(s)
    #matrix multiplication   O(n**3)
    def __mul__(self,a):
        if type(a)!=mtx:
            s=[];
            for i in range(0,self.c):
                si=[]
                for j in range(0,self.r):
                    si+=[a*self.m[i][j]]
                s+=[si]
            return mtx(s)
        at=a.transpose()
        s=[];
        for i in range(0,self.c):
            si=[]
            for j in range(0,at.c):
                si+=[dot(self.m[i],at.m[j])]
            s+=[si]
        return mtx(s)
    #A/B = A*inv(B) where A and B are square matrices  O(n**3)
    def __truediv__(self,a):
        if type(a)!=mtx:
            s=[];
            for i in range(0,self.c):
                si=[]
                for j in range(0,self.r):
                    si+=[self.m[i][j]/a]
                s+=[si]
            return mtx(s)
        return self*a.inv()
    #raise a matrix to the power of an integer  O(abs(x)*n**3)
    def __pow__(self,x):
#        s=[];
#        for i in range(0,self.c):
#            si=[]
#            for j in range(0,self.r):
#                if i==j: si+=[1]
#                else: si+=[0]
#            s+=[si]
#        I=mtx(s)
        I=identity(self.c)
        a=self
        if x<0: a=self.inv()
        for i in range(0,abs(x)):
            I=I*a
        return I
    #matrix scalar multiplication is communitive
    def __rmul__(self,p):
        if type(p)==mtx: raise NameError('')
        return self*p
    #implementing +=,-=,*= ...
    def __iadd__(self,p):
        self=self+p
    def __isub__(self,p):
        self=self-p
    def __imul__(self,p):
        self=self*p
    def __idiv__(self,p):
        self=self/p
    def __ipow__(self,p):
        self=self**p
    #transpose  O(n**2)
    def transpose(self):
        s=[];
        for i in range(0,self.r):
            si=[]
            for j in range(0,self.c):
                si+=[self.m[j][i]]
            s+=[si]
        return mtx(s)
    #create a copy  O(n**2)
    def copy(self):
        s=[];
        for i in range(0,self.c):
            si=[]
            for j in range(0,self.r):
                if type(self.m[i][j])==poly:
                    si+=[poly(self.m[i][j].c)]
                else: si+=[self.m[i][j]]
            s+=[si]
        return mtx(s)
    def factor(self,A,N,Tol,P):
        for i in range(0,N):
            # print(mtx(A))
            maxA=0.0
            # if type(A[0][0])==poly: maxA=poly([0])
            imax=i
            for k in range(i,N):
                # if type(A[k][i])==None: A[k][i]=poly([0])
                # print(type(A[k][i]))
                absA=abs(A[k][i])
                if absA>maxA:
                    maxA=absA
                    imax=k
            if maxA<Tol: return False
            if imax!=i:
                j=P[i]
                P[i]=P[imax]
                P[imax]=j
                ptr=A[i]
                A[i]=A[imax]
                A[imax]=ptr
                P[N]+=1
            for j in range(i+1,N):
                A[j][i]/=A[i][i]
                for k in range(i+1,N):
                    A[j][k]-=A[j][i]*A[i][k]
        return True;
    def inv(self):
        if self.c!=self.r: raise NameError('not square')
        A=self.copy().m
        N=self.c
        P=[]
        for i in range(0,N+1):
            P+=[i];
        IA=[]
        for j in range(0,N):
            IAi=[]
            for i in range(0,N):
                IAi+=[0]
            IA+=[IAi]
        self.factor(A,N,.000001,P)
        for j in range(0,N):
            for i in range(0,N):
                if P[i]==j:
                    IA[i][j]=1.0
                else:
                    IA[i][j]=0.0
                for k in range(0,i):
                    IA[i][j]-=A[i][k]*IA[k][j]
            for i in range(N - 1,-1,-1):
                for k in range(i+1,N):
                    IA[i][j]-=A[i][k]*IA[k][j]
                IA[i][j]=IA[i][j]/A[i][i]
        return mtx(IA)
    def det(self):
        if self.c!=self.r: raise NameError('not square')
        A=self.copy().m
        # print(mtx(A))
        N=self.c
        P=[]
        for i in range(0,N+1):
            P+=[i];
        self.factor(A,N,.000001,P)
        det=A[0][0]
        for i in range(1,N):
            det*=A[i][i]
        if (P[N]-N)%2==0: return det;
        else: return -det;
    #convert the matrix to a numpy array
    #this is used to validate to correctness of these algorithms
    #by comparing them to the numpy versions
    def to_np(self):
        return np.array(self.m)
    #stringify the matrix so it print()s nicely
    def __str__(self):
        s='['
        for i in range(0,self.c):
            s+='['
            for j in range(0,self.r):
                try:
                    s+=str(np.round(self.m[i][j],5))
                except:
                    s+=str(self.m[i][j])
                if j<self.r-1: s+=','
            s+=']'
            if i<self.c-1: s+=',\n '
        s+=']\n'
        return s
    #extract the minor matrix of given element (necessary for finding the cofactor of an element)
    #O(n**2)
    def getM(self,k,l):
        if (not 0<=k<self.c) or (not 0<=l<self.r): raise NameError('index out of range')
        M=self.copy()
        del M.m[k]
        for i in range(0,len(M.m)):
            del M.m[i][l]
        M.c-=1
        M.r-=1
        return M
    #cofactor is necessary for fining the adjoint matrix  O(n**3)
    def cofactor(self,k,l):
        return self.getM(k,l).det()*(-1)**(k+l)
    #finding the determinant recursively  O(n!) ... yikes
    def det2(self):
        if self.c!=self.r: raise NameError('not square')
        if self.c==1: return self.m[0][0]
        det=0
        if type(self.m[0][0])==poly: det=poly([0])
        for i in range(0,self.c):
            det=det+self.getM(i,0).det2()*self.m[i][0]*(-1)**i
        return det
    #finding the adjoint matrix (necessary for finding the inverse) O(n**5)
    def adj(self):
        if self.c!=self.r: raise NameError('not square')
        s=ones(self.c,self.r)*0
        for i in range(0,self.c):
            for j in range(0,self.r):
                s.m[j][i]=self.cofactor(i,j)
        return s
    #second methed of finding the inverse  O(n**5)
    def inv2(self):
        return self.adj()/self.det()
    #finding the charactarist polynomial for the purpose of finding eigenvalues
    #O(n**3)
    def charpoly(self):
        if self.c!=self.r: raise NameError('not square')
        p=self.copy()
        for i in range(0,self.c):
            for j in range(0,self.c):
                if i==j: p.m[i][j]=poly([p.m[i][j],-1])
                else: p.m[i][j]=poly([p.m[i][j]])
        return p.det3()
    #one way of quantifying the size of matrix (mostly used to determine
    #if the matrix has all elements close to zero but not exactly)  O(n**2)
    def mag(self):
        s=0
        for i in range(0,self.c):
            for j in range(0,self.r):
                s+=self.m[i][j]**2
        return s**.5
    #extract a column
    def col(self,n):
        a=[]
        for i in range(0,self.c): a+=[[self.m[i][n]]]
        return mtx(a)
    #extract a row
    def row(self,n):
        a=[]
        for i in range(0,self.r): a+=[self.m[n][i]]
        return mtx([a])
    #finding the eigenvalues
    def eigenvals(self):
        return self.charpoly().roots()
    #finding the eigenvectors
    def eigenvecs(self):
        if self.c!=self.r: raise NameError('not square')
        vals=self.eigenvals()
        vecs=[]
        A=self.getM(0,0)
        for k in range(0,len(vals)):
            M=A-identity(self.c-1)*vals[k]
            b=[]
            for i in range(1,self.c): b+=[[-self.m[i][0]]]
            b=mtx(b)
            v=M.inv()*b
            v=mtx([[1]+v.transpose().m[0]])
            v=v*-1
            v=v/v.mag()
            vecs+=v.m
        vecs=mtx(vecs).transpose()
        for i in range(0,self.c):
            if abs((self*vecs.col(i)-vecs.col(i)*vals[i]).mag())>.001:
                raise NameError('invalid eigen vector')
        #normalize the vectors
        for i in range(0,vecs.c):
            colmag=0;
            for j in range(0,vecs.r):
                colmag+=vecs.m[j][i]**2
            colmag=colmag**.5
            for j in range(0,vecs.r):
                vecs.m[j][i]/=colmag
        #verify that the vectors are correct
        for i in range(0,self.c):
            if abs((self*vecs.col(i)-vecs.col(i)*vals[i]).mag())>.00001:
                raise NameError('invalid eigen vector')
        return vecs
    #add a times row i to row j (necessary for guassian elimination)  O(n)
    def addAItoJ(self,a,i,j):
        for k in range(0,self.r): self.m[j][k]=self.m[j][k]+a*self.m[i][k]
    #swap two matrix rows  O(1) because were swapping pointers
    def swap(self,i,j):
        if i==j: return
        tmp=self.m[i]
        self.m[i]=self.m[j]
        self.m[j]=tmp;
    #multiply each element in row i by scalar a (necessary for guassian elimination)
    #O(n)
    def rowmul(self,i,a):
        for j in range(0,self.r): self.m[i][j]=self.m[i][j]*a
    #finding the determinant of a matrix via gaussian elimination O(n**3)
    def det3(self):
        if self.c!=self.r: raise NameError('not square')
        A=self.copy()
        sgn=1
        pt=False
        if type(self.m[0][0])==poly: pt=True
        if pt: sgn=poly([1])
        for i in range(0,self.c):
            if A.m[i][i]==0:
                b=False
                for j in range(i+1,self.c):
                    if A.m[j][i]!=0:
#                        print(sgn)
                        sgn=sgn*-1
#                        print(sgn,'\n')
                        A.swap(i,j)
                        b=True
                        break
                if not b: return 0
            for j in range(i+1,self.c):
                sgn=sgn*A.m[i][i]
                tmp=A.m[j][i]
                A.rowmul(j,A.m[i][i])
                A.addAItoJ(-tmp,i,j)
        d=1;
        if pt: d=poly([1])
        for i in range(0,self.c): d=d*A.m[i][i]
#        print(sgn)
        return d/sgn
    #finding the inverse of a matrix via gaussian elimination  O(n**3) nice
    def inv3(self):
        if self.c!=self.r: raise NameError('not square')
        A=self.copy()
        I=identity(self.r)
        for i in range(0,self.r):
            A.m[i]+=I.m[i]
        A.r*=2
        for i in range(0,A.c):
            if A.m[i][i]==0:
                b=False
                for j in range(i+1,A.c):
                    if A.m[j][i]!=0:
                        A.swap(i,j)
                        b=True
                        break
                if not b: raise NameError('uninvertible')
            for j in range(i+1,A.c):
                A.addAItoJ(-A.m[j][i]/A.m[i][i],i,j)
        for i in range(A.c-1,0,-1):
            if A.m[i][i]==0:
                b=False
                for j in range(A.c-1,i,-1):
                    if A.m[j][i]!=0:
                        A.swap(i,j)
                        b=True
                        break
                if not b: raise NameError('uninvertible')
            for j in range(i-1,-1,-1):
                A.addAItoJ(-A.m[j][i]/A.m[i][i],i,j)
        for i in range(0,A.c):
            for j in range(A.r-1,self.c-1,-1):
                A.m[i][j]/=A.m[i][i]
        for i in range(0,self.c):
            A.m[i]=A.m[i][self.c:]
        A.r//=2
        return A

#polynomial class
#a polynomial can be represented as a vector of coefficients, for example
#the polynomial a + b*x + c*x**2 + d*x**3
#can be represented as [a,b,c,d]
#this class is used to find eigenvalues
class poly:
    def __init__(self,c):
        self.c=[]
        for i in c: self.c+=[i];
        if len(c)==0: self.c=[0]
        for i in range(len(c)-1,0,-1):
            if self.c[i]==0:
                del self.c[i]
            else: return
    #polynomial vector addition is equivalent to normal vector addition O(n)
    def __add__(self,p):
        if type(p)!=poly: p=poly([p])
        s=[]
        for i in range(0,max(len(self.c),len(p.c))):
            s+=[0]
            if i<len(self.c): s[i]+=self.c[i]
            if i<len(p.c): s[i]+=p.c[i]
        return poly(s)
    #same with subtraction  O(n)
    def __sub__(self,p):
        if type(p)!=poly: p=poly([p])
        s=[]
        for i in range(0,max(len(self.c),len(p.c))):
            s+=[0]
            if i<len(self.c): s[i]+=self.c[i]
            if i<len(p.c): s[i]-=p.c[i]
        return poly(s)
    #FOIL! say A=B*C where A,B,C are polynomials and A_c is the coefficent
    #vector that represents A such that A = sum_over_i( A_c[i]*x**i )
    #C = sum_over_i( sum_over_j( A_c[i]*B_c[j]*x**(i+j) ))
    #O(n**2)
    def __mul__(self,p):
        if type(p)!=poly: p=poly([p])
        s=[];
        for i in range(0,len(self.c)+len(p.c)-1): s+=[0];
        for i in range(0,len(self.c)):
            for j in range(0,len(p.c)):
                s[i+j]+=self.c[i]*p.c[j]
        return poly(s)
    #polynomial long division  ... remember this from algebra class ? .___.
    #O(n(m-n+1))
    def __truediv__(self,p):
        if type(p)!=poly: p=poly([p])
        if len(p.c)==0 or (len(p.c)==1 and p.c[0]==0): raise ZeroDivisonError('')
        if len(self.c)<len(p.c):
            print('bruh')
            return poly([0])
        s=poly(self.c);
        r=[]
        for i in range(0,len(self.c)): r+=[0]
        for i in range(len(self.c)-len(p.c),-1,-1):
            n=s.c[i+len(p.c)-1]/p.c[-1]
            for j in range(0,len(p.c)):
                s.c[i+j]-=n*p.c[j]
            r[i+j-len(p.c)+1]=n
        return poly(r)
    #same as division except we return the remaineder instead of the quotient
    def __mod__(self,p):
        if type(p)!=poly: p=poly([p])
        if len(p.c)==0 or (len(p.c)==1 and p.c[0]==0): raise ZeroDivisonError('')
        if len(self.c)<len(p.c):
            print('bruh')
            return poly([0])
        s=poly(self.c);
        r=[]
        for i in range(0,len(self.c)): r+=[0]
        for i in range(len(self.c)-len(p.c),-1,-1):
            n=s.c[i+len(p.c)-1]/p.c[-1]
            for j in range(0,len(p.c)):
                s.c[i+j]-=n*p.c[j]
            r[i+j-len(p.c)+1]=n
        return poly(s.c)
    #raise a polynomial to the power of an integer  O(x*n**2)
    def __pow__(self,x):
        if(x<0): raise NameError('negative power')
        p=poly([1])
        for i in range(0,x):
            p=p*self
        return p
    #some operators for convinience
    def __neg__(self):
        return self*-1
    def __iadd__(self,p):
        self=self+p
    def __isub__(self,p):
        self=self-p
    def __imul__(self,p):
        self=self*p
    def __idiv__(self,p):
        self=self/p
    def __ipow__(self,p):
        self=self**p
    def __lt__(self,p):
        if type(p)!=poly: p=poly([p])
        return self.c[0]<p.c[0]
    def __gt__(self,p):
        if type(p)!=poly: p=poly([p])
        return self.c[0]>p.c[0]
    def __eq__(self,p):
        return len(self)<=1 and self.c[0]==p
    def __neq__(self,p):
        return not self==p
        # if type(p)!=poly: p=poly([p])
        # b=True
        # for i in self.c[==p.c[0]
    def __abs__(self):
        # s=poly(self.c);
        # for i in range(0,len(self.c)): s.c[i]=abs(s.c[i])
        return poly([1,1])
    def __len__(self):
        return len(self.c)
    #calculus time.  There is no general formula for finding the roots
    #of a polynomial with order > 5 so we must use numerical analysis
    # say we want to find x such that f(x)=0
    # let x[0] be a guess
    #if we follow the update rule:
    #    x[n]=f(x[n-1])/( df/dx (x[n]) )
    #iteratively, then x[n] will rapidily approach x
    #this is called Newton's Method
    def newton(self,x=1+1j):
        # x=np.random.random()+np.random.random()*1j
        for i in range(0,100):
            x-=self.compute(x)/self.diff().compute(x)
        if abs(self.compute(x))>.001: raise NameError('did not converge')
        return x
    def roots(self):
        v=[]
        self.rts(v)
        return v
    #use newtons method to find a root r, then factor out that root with
    #long division, then find the next root, factor it out and so on.
    def rts(self,v):
        if len(self.c)<=1: return
        r=self.newton()
        v+=[r]
        q=poly([-r,1]);
        n=self/q
        if len(n.c)==len(self.c): del n.c[-1]
        n.rts(v)
    #more calculus
    #using the power rule to compute the derivative of a polynomial O(n)
    def diff(self):
        s=[]
        for i in range(1,len(self.c)): s+=[i*self.c[i]]
        return poly(s)
    #finding the value of a polynomial at a point x  O(n)
    def compute(self,x):
        s=0
        for i in range(0,len(self.c)): s+=self.c[i]*x**i
        return s
    #stringify so it print()s nicely
    def __str__(self):
        s=''
        for i in range(0,len(self.c)):
            s+=str(round(self.c[i],5))
            s+='*x^'
            s+=str(i)
            if i<len(self.c)-1:s+=' + '
        return s

#generates an identity matrix
def identity(n):
    s=[]
    for i in range(0,n):
        si=[]
        for j in range(0,n):
            if i==j: si+=[1]
            else:    si+=[0]
        s+=[si]
    return mtx(s)
    
#generates a matrix of all ones
def ones(m,n):
    s=[]
    for i in range(0,m):
        si=[]
        for j in range(0,n):
            si+=[1]
        s+=[si]
    return mtx(s)

def cramer(A,b):
    s=[]
    Adet=A.det()
    for i in range(0,b.c):
        Ai=A.transpose()
        Ai.m[i]=b.transpose().m[0]
        Ai=Ai.transpose()
        s+=[[Ai.det()/Adet]]
    return mtx(s)



n=3; #length and width of the matrix

#generate random matices
Anp=2*(np.random.random([n,n])-.5)
Bnp=2*(np.random.random([n,n])-.5)
A=mtx(Anp);
B=mtx(Bnp);

print("to test the algorithms, Im comparing their output to numpy's output")

#testing arithmatic, inversion, and detemination
for i in range(0,20):
    if i==0:
        s="testing A+B"
        C=A+B         #my implementation
        Cnp=Anp+Bnp   #numpy implementation
    elif i==1:
        s="testing A-B"
        C=A-B
        Cnp=Anp-Bnp
    elif i==2:
        s="testing A*B"
        C=A*B
        Cnp=np.dot(Anp,Bnp)
    elif i==3:
        s="testing A.inv()"
        C=A.inv()
        Cnp=np.linalg.inv(Anp)
    elif i==4:
        s="testing A.inv2()"
        C=A.inv2()
        Cnp=np.linalg.inv(Anp)
    elif i==5:
        s="testing A.inv3()"
        C=A.inv3()
        Cnp=np.linalg.inv(Anp)
    elif i==6:
        s="testing A.det()"
        C=A.det()
        Cnp=np.linalg.det(Anp)
    elif i==7:
        s="testing A.det2()"
        C=A.det2()
        Cnp=np.linalg.det(Anp)
    elif i==8:
        s="testing A.det3()"
        C=A.det3()
        Cnp=np.linalg.det(Anp)
    else: break
    diff=0
    if type(C)==mtx:
        diff_mtx=C.to_np()-Cnp
        diff=mtx(diff_mtx).mag()
    else:
        diff=C-Cnp
    diff=abs(diff)
    print(s)
    print(f"\tdifference = {diff}")
    if diff>.00001: print("\tfailed")
    else: print("\tpassed")

print("testing the eigenvalues of A")
eig=np.linalg.eig(Anp)
vals=np.array(A.eigenvals())
npvals=eig[0]

#the eigenvalues are not ordered so we mush compare them to find a match
for i in range(0,n):
    match=1e10
    for j in range(0,n):
        match=min(match,abs(abs(vals[i])-abs(npvals[j])))
    diff+=match
        

#print(vals)
#print(eig[0])
print(f"\tdifference = {diff}")
if diff>.00001: print("\tfailed")
else: print("\tpassed")



print("testing the eigenvectors of A")
vecs=A.eigenvecs().to_np().T
npvecs=eig[1].T

#print(mtx(vecs))
#print(mtx(npvecs))
#the eigenvectors are not ordered so we mush compare them to find a match
diff=0
for i in range(0,n):
    match=1e10
    for j in range(0,n):
        match=min( match,sum(abs(abs(vecs[i]) - abs(npvecs[j])) ))
    diff+=match
        

print(f"\tdifference = {diff}")
if diff>.00001: print("\tfailed")
else: print("\tpassed")










