import numpy as np
import matplotlib.pyplot as plt
import math
#secant method
def secant_method(func, x0, x1, tol=1e-6, max_iter=50):
    iteration = 0
    while iteration < max_iter:
        f_x0 = func(x0)
        f_x1 = func(x1)
        if abs(f_x1) < tol:
            return x1 
        elif abs(f_x0) < tol:
            return x0
        else:
            x_next = x1 - (f_x1 * (x1 - x0)) / (f_x1 - f_x0)
            if abs(x_next - x1) < tol:
                return x_next
            x0, x1 = x1, x_next
            iteration += 1
    return x1
#bisection method
def bisection_method(a,b,func,ep=pow(10,-6),delt=pow(10,-9)): #solves using bisection method   
    c = a
    solulis,error=[],[]
    while ((b-a) >= ep):        
        c = (a+b)/2        
        if (abs(func(c))<delt):#if f(c) is close to 0,break loop
            break        
        if (func(c)*func(a) < 0):
            b = c
        else:
            a = c
        error.append(abs(b-a))
        solulis.append(c)
    return(c,solulis,error)#c is the final root, solulis is all the values of c and error is error for each c
#fixed point iteration
def fixed_point_iteration(f, initial_guess, tolerance=1e-4, max_iterations=100):
    x = initial_guess
    iteration = 0    
    while iteration < max_iterations:
        x_new = f(x)
        curr_error=abs(x_new - x)
        if curr_error < tolerance:
            return x_new
        x = x_new
        iteration += 1    
    print("Solution not converging, change tolerance or max_iteration.")
    print(f"Current error {curr_error} after {max_iteration} iterations.")
    return None
# Simpson's Rule
def simpsonwithn(a,b,f,n,ep=pow(10,-6)):#integration by simpson method with a predefined number of steps
    h=(b-a)/n                                   
    sum1=f(a)
    for i in range(2,n,2):
        sum1+=2*f(a+i*h)
    for j in range(1,n,2):
        sum1+=4*f(a+j*h)
    sum1+=f(b)
    return(h*sum1/3)
def returnnsimpson(a,b,ep,trim):#finds n for simpson method
    x=pow(b-a,5)*trim/ep
    n=x/180
    return math.ceil(pow(n,0.25))
def simpson(a,b,f,ep,trim):#integration by simpson method
    n=returnnsimpson(a,b, ep, trim)
    print("N: ",n)
    h=(b-a)/n                                   
    sum1=f(a)
    for i in range(2,n,2):
        sum1+=2*f(a+i*h)
    for j in range(1,n,2):
        sum1+=4*f(a+j*h)
    sum1+=f(b)
    return(h*sum1/3)
#midpoint and trapezoidal and comparision
def midpoint(a,b,n,f):
    sum1=0
    h=(b-a)/n
    x=a+(h/2)
    for i in range(n):
        sum1+=f(x)
        x+=h
    return h*sum1
def interpolation(xp,x,y):
    n=len(x)
    sums=0
    for i in range(n):
        p=1
        for j in range(n):
            if i!=j:
                p=p*((xp-x[j])/(x[i]-x[j]))
        sums=sums+p*y[i]
    return(sums)
def trapezoidal(a,b,n,f):
    sum1=0
    h=(b-a)/n
    x=a
    for i in range(1,n):
        x+=h
        sum1+=f(x)
    sum1=sum1*2
    sum1+=f(a)+f(b)
    return (h/2)*sum1
def MiTrSi(f,a,b,steps=[10,20,30]):
    mi,tr,si=[],[],[]
    for i in steps:
        mi.append(midpoint(a, b, i, f))
        tr.append(trapezoidal(a,b, i, f))
        si.append(simpsonwithn(a, b, f,i ))
    return mi,tr,si
def returnnsimpson(a,b,ep,trim):#finds n for simpson method, trim is max of 4th derivative
    x=pow(b-a,5)*trim/ep
    n=x/180
    return math.ceil(pow(n,0.25))
def returnnmidpoint(a,b,ep,trim):#finds n for midpoint method, trim is max of 2nd derivative
    x=pow(b-a,3)*trim/ep
    n=x/24
    return math.ceil(pow(n,0.5))
def returnntapezoidalpoint(a,b,ep,trim):#finds n for trapezoidal method, trim is max of 2nd derivative
    x=pow(b-a,3)*trim/ep
    n=x/12
    return math.ceil(pow(n,0.5))
#regula falsi
def regula_falsi(f,a,b,eps):#solves using regula falsi method in interval a to b
    count=0
    solulis,error=[],[]
    if f(a)*f(b)<0:
        if abs(b-a)<eps :
            return b,count,solulis,error
        while abs(b-a)>eps:            
            c=b-(((b-a)*f(b))/(f(b)-f(a)))
            if f(a)*f(c)<0:
                b=c
            if abs(f(c))<eps:
                return c,count,solulis,error #returns solution,no of steps, values of c,error
            if f(b)*f(c)<0:
                a=c
            error.append(abs(((b-a)*f(b))/(f(b)-f(a))))
            solulis.append(c)
            count+=1
#forward euler
def forward_euler(f,x,y,xn,h=0.005):
    n=math.ceil((xn-x)/h)
    for i in range(n):
        y=y+h*f(x,y)
        x=x+h
    return y
#backward euler
def backward_euler(func, dfunc_dt, y0, T, dt):
    num_steps = int(T / dt)
    time_points = [i * dt for i in range(num_steps + 1)]

    y_values = [y0]
    for t in time_points[1:]:
        y_new = newton_raphson_backward(func, dfunc_dt, t, y0, dt)
        y_values.append(y_new)
        y0 = y_new
    return time_points, y_values

def newton_raphson_backward(func, dfunc, t, initial_guess, dt, ep=1e-6):
    ctr1 = 0
    err, solu_lis = [], []
    temp = func(t, initial_guess) / dfunc(t, initial_guess)
    while abs(temp) > ep:
        ctr1 += 1
        temp = func(t, initial_guess) / dfunc(t, initial_guess)
        err.append(temp)
        initial_guess -= temp
        solu_lis.append(initial_guess)
    return initial_guess
#newton raphson
def newton_raphson(initial_guess,func,dervi,ep=pow(10,-6)):
    ctr1=0
    err,solu_lis=[],[]
    temp=func(initial_guess)/dervi(initial_guess)
    while abs(temp)>ep:
        ctr1+=1
        temp=func(initial_guess)/dervi(initial_guess)
        err.append(temp)
        initial_guess-=temp
        solu_lis.append(initial_guess)
    return initial_guess,ctr1,solu_lis,err#returns solution(initial guess becomes solution),counter, list of solution guess,error by newton raphson method

#Gaussian Quadrature 
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)
def legendre_coeffes(n):#get the coefficeints of the legrande polynomail
    coeff_leg=[0]*(n+1)
    if n%2==0:
        k=int(n/2)
    else:
        k=int((n-1)/2)
    for i in range(0,k+1):#k+1 as for loop (0,n) stops at n-1
        coeff_leg[n-2*i]=((-1)**(i))*(factorial(2*n-2*i)/((2**(n))*factorial(i)*factorial(n-i)*factorial(n-2*i)))
        #regular formula for legendre polynomial(non recursive) http://hyperphysics.phy-astr.gsu.edu/hbase/Math/legend.html
    coeff_leg=coeff_leg[::-1]#reverses the array as the 0th index is currently highest x power
    return coeff_leg
def roots_of_legendre(n):#get roots of the nth order legrance polynomial
    co=legendre_coeffes(n)
    #print("Coeffes of legendre:",co)    
    roots= polyRoots(co)
    for i in range (0,len(roots)):
        roots[i]=round(roots[i],6)#tolerance of poly roots is 10^(-6) so rounding to 6
    return roots
def get_weights_legendre(roots):#get weights of eqch root
    n=len(roots)#n is the degree of legendre polynomial
    weight=[0 for _ in range(n)]
    for i in range (0,len(roots)):
        a=legendre_coeffes(n)        
        weight[i]=2/((1-(roots[i])**2)*(deriv(a,roots[i])**2))#https://www.thermopedia.com/content/811/
        weight[i]=round(weight[i],6)#as roots[i] is precise only upto 10^(-6)
    return weight
def gaussian_quadrature_legendre(integrand,n):
    roots=roots_of_legendre(n)
    weights=get_weights(roots)
    sum1=0
    for i in range(len(roots)):
        sum1=sum1+(weights[i]*integrand(roots[i]))
    return sum1
#gaussian_quadrature_Lagurre
def roots_of_Lagurre(n):#get roots of the nth order legrandre polynomial
    co=Lagurre_coeffes(n)
    #print("Coeffes of Lagurre:",co)    
    roots= PolynomialRoots(co).roots
    for i in range (0,len(roots)):
        roots[i]=round(roots[i],6)#tolerance of poly roots is 10^(-6) so rounding to 6
    return roots
def Lagurre_coeffes(n):#get the coefficeints of the Lagurre polynomial
    coeff_Lagurre=[0]*(n+1)
    for i in range(0,n+1):#k+1 as for loop (0,n) stops at n-1
        coeff_Lagurre[i]=((-1)**(i))*(factorial(n)/((factorial(i)**2)*factorial(n-i)))
    coeff_Lagurre=coeff_Lagurre[::-1]#reverses the array as the 0th index is currently highest x power
    return coeff_Lagurre
def get_weights_Lagurre(roots):#get weights of eqch root
    n=len(roots)#n is the degree of legendre polynomial
    Lnplus1=Lagurre_coeffes(n+1)
    weight=[0 for _ in range(n)]
    for i in range (0,len(roots)):
        weight[i]=roots[i]/(((n+1)**2)*((solvepoly(Lnplus1,roots[i]))**2))#https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature
        weight[i]=round(weight[i],6)#as roots[i] is precise only upto 10^(-6)
    return weight
def gaussian_quadrature_Lagurre(integrand,n):
    roots=roots_of_Lagurre(n)
    weights=get_weights_Lagurre(roots)
    sum1=0
    for i in range(len(roots)):
        sum1=sum1+(weights[i]*integrand(roots[i]))
    return sum1
#gaussian_quadrature_Hermite
def roots_of_Hermite(n):#get roots of the nth order legrandre polynomial
    co=Hermite_coeffes(n)
    #print("Coeffes of Hermite:",co)    
    roots= PolynomialRoots(co).roots
    for i in range (0,len(roots)):
        roots[i]=round(roots[i],6)#tolerance of poly roots is 10^(-6) so rounding to 6
    return roots
def Hermite_coeffes(n):#get the coefficeints of the Hermite polynomial
    coeff_Hermite=[0]*(n+1)
    if n%2==0:
        k=int(n/2)
        for i in range(0,k+1):#k+1 as for loop (0,n) stops at n-1
            coeff_Hermite[2*i]=factorial(n)*((-1)**(k-i))*((2**(2*i))/(factorial(2*i)*factorial(k-i)))
    else:
        k=int((n-1)/2)
        for i in range(0,k+1):#k+1 as for loop (0,n) stops at n-1
            coeff_Hermite[2*i+1]=factorial(n)*((-1)**(k-i))*((2**(2*i+1))/(factorial(2*i+1)*factorial(k-i)))
    coeff_Hermite=coeff_Hermite[::-1]#reverses the array as the 0th index is currently highest x power
    return coeff_Hermite
def get_weights_Hermite(roots):#get weights of eqch root
    n=len(roots)#n is the degree of legendre polynomial
    Hnminus1=Hermite_coeffes(n-1)
    weight=[0 for _ in range(n)]
    for i in range (0,len(roots)):
        weight[i]=2**(n-1)*factorial(n)*(np.pi**(0.5))/((n**2)*((solvepoly(Hnminus1,roots[i]))**2))#https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
        weight[i]=round(weight[i],6)#as roots[i] is precise only upto 10^(-6)
    return weight
def gaussian_quadrature_Hermite(integrand,n):
    roots=roots_of_Hermite(n)
    weights=get_weights_Hermite(roots)
    sum1=0
    for i in range(len(roots)):
        sum1=sum1+(weights[i]*integrand(roots[i]))
    return sum1
#RK4
def rk4(x0,y0,x,h,f):
    n=round((x-x0)/h)
    y=y0
    xlis=[]
    ylis=[]
    for i in range(1,n+1):
        k1=h*f(x0,y)
        k2=h*f(x0+h/2,y+k1/2)
        k3=h*f(x0+h/2,y+k2/2)
        k4=h*f(x0+h,y+k3)
        y=y+((k1+2*k2+2*k3+k4)/6)
        x0=x0+h
        xlis.append(x0)
        ylis.append(y)
    return y
#predictor corrector
def predictorcorrector(f,x0,y0,xn,h=0.005):
    x=x0
    y=y0
    n=math.ceil((xn-x)/h)
    for i in range(n):
        k1=h*f(x,y)
        k2=h*f(x+h,forward_euler(f, x0, y0, x+h))
        y=y+(k1+k2)/2
        x=x+h
    return y

#monte carlo integration
def montecarlo(f,a,b,n,seed=10):
    lisran=[]
    for i in range(n):
        seed=LCG(seed)
        lisran.append(a+(seed*(b-a)/32768))
    intergra=0
    for ran in lisran:
        intergra+=f(ran)
    return ((b-a)*intergra/n)
def LCG(seed):
    a = 1664525
    c = 1013904223
    m = 32768
    seed = (a * seed + c) % m
    return seed
#Laplace Equation five point calculation
def five_point_laplace(u,Lx,Ly,nx,ny):
    # Make the grid
    x_values = np.linspace(0, Lx, nx)
    y_values = np.linspace(0, Ly, ny)
    dx, dy = Lx / (nx + 1), Ly / (ny + 1)
    X = [[x] * len(y_values) for x in x_values]
    Y = [y_values] * len(x_values)
    max_iter = 1000
    tolerance = 1e-6
    h_squared = dx * dy  
    for k in range(max_iter):
        u_old = u.copy()
        for i in range(1, nx -1):
            for j in range(1, ny-1 ):
                u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1]) 
        if np.linalg.norm(u - u_old) < tolerance:#till tolerance is reached dont break
            break
    return u
#poissons Equation five point calculation
def five_point_poisson(u,Lx,Ly,nx,ny,psi):
    # Make the grid
    x_values = np.linspace(0, Lx, nx)
    y_values = np.linspace(0, Ly, ny)
    dx, dy = Lx / (nx + 1), Ly / (ny + 1)
    X = [[x] * len(y_values) for x in x_values]
    Y = [y_values] * len(x_values)
    rho = np.zeros_like(X)#This accounts for different len(y_values) and len(x_values); which is not the case here but just in case(as this is a modification of my code from DIY project(for laplacian) and I used differnet lenghts for meshgrid there)
    for i in range(nx):
        for j in range(ny):
            rho[i][j] = psi(X[i][j],Y[i][j])
    max_iter = 1000
    tolerance = 1e-6
    h_squared = dx * dy  
    for k in range(max_iter):
        u_old = u.copy()
        for i in range(1, nx -1):
            for j in range(1, ny-1 ):
                u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1]) + h_squared * rho[i][j]/4
        if np.linalg.norm(u - u_old) < tolerance:#till tolerance is reached dont break
            break
    return u
#shooting method
import matplotlib.pyplot as plt
def rk4_shoot(dx,dv,t,T,dt,x0,v0):    
    n=round((T-t)/dt)
    x=x0
    xlis=[x0]
    vlis=[v0]
    tlis=[t]
    v=v0
    for i in range(1,n+1):
        k1x=dt*dx(x,v,t)
        k1v=dt*dv(x,v,t)
        k2x=dt*dx(x+k1x/2,v+k1v/2,t+dt/2)
        k2v=dt*dv(x+k1x/2,v+k1v/2,t+dt/2)
        k3x=dt*dx(x+k2x/2,v+k2v/2,t+dt/2)
        k3v=dt*dv(x+k2x/2,v+k2v/2,t+dt/2)
        k4x=dt*dx(x+k3x,v+k3v,t+dt)
        k4v=dt*dv(x+k3x,v+k3v,t+dt)
        t+=dt
        x+=(k1x+2*k2x+2*k3x+k4x)/6
        v+=(k1v+2*k2v+2*k3v+k4v)/6
        xlis.append(x)
        vlis.append(v)
        tlis.append(t)
    return x,v,xlis,vlis,tlis
def shootingmethod(rk4,dv,dx,guess1,guess2,x0,xmax,t,T,dt,tol=pow(10,-3)):
    varr,xarr,tarr=[],[],[]
    x1,v1,*rest=rk4(dx, dv, t, T, dt, x0, guess1)
    x2,v2,*rest=rk4(dx,dv,t, T, dt,x0,guess2)
    itera=0
    while abs(x1-xmax)>tol and abs(x2-xmax)>tol :
        itera+=1  
        vnew=v2+((xmax-x2)*(v1-v2))/(x1-x2)
        xnew,vnew2,*temp=rk4(dx, dv, t, T, dt, x0, vnew)
        if abs(xmax-xnew)<1:
            xf,vf,xarr,varr,tarr=rk4(dx, dv, t, T, dt, x0, vnew)
            break
        else:
            if xnew<xmax:
                x2=xnew
                v2=vnew
            else:
                x1=xnew
                v1=vnew
    plt.plot(tarr,xarr,'r--',label="T")
    plt.plot(tarr,varr,'b--',label='dT/dx')
    plt.legend()
    plt.axvline(x = 4.4)
    plt.axhline(y =100)
    plt.xlabel("L")
    plt.savefig("q2.png")
    plt.show()
    p=0
    for i in range(len(xarr)):
        if abs(xarr[i]-100)<0.1:
            p=i
    return xarr,tarr,p

#Du Fort method
def DuFort(xmax, nx,tmax, nt,tplot):
    V = []
    hx = xmax/nx
    ht = tmax/nt
    X = []
    x0 = 0
    for i in range(0, nx+1):
        if x0 + (hx*i) == 1:
            V.append(300)
        else:
            V.append(0)
        X.append(x0 + hx*i)
    a = ht/(hx*hx)
    print("Stability Condition :",a)
    for j in range (0, nt):
        temp = []
        if j in tplot:
            plt.plot(X, V, label = str(j))            
        for k in range (0, nx+1):
            if k == 0:
                p = (1-2*a)*V[k] + a*V[k+1]
            elif k == nx:
                p = a*V[k-1] + (1-2*a)*V[k]
            else:
                p = a*V[k-1]+(1-2*a)*V[k]+a*V[k+1]
            temp.append(p)
        for q in range(0, len(V)-1):
            V[q] = temp[q]
    plt.legend(loc='right')
    plt.xlabel("Distance x")
    plt.ylabel("Temperature T")
    plt.show()
    return None
#Crank Nicolson method
def crank_nicolson_solver(u_prev, B, identity_matrix, alpha):
    Nx = len(u_prev)    
    M1 = Add_matrices(identity_matrix, multiply_matrix_by_constant(B,alpha))
    M2 = Subtract_matrices(identity_matrix, multiply_matrix_by_constant(B,alpha))
    M1_inv=np.linalg.inv(M1)
    return matrix_mul_B_is_1D(M1_inv, matrix_mul_B_is_1D(M2, u_prev))
def crank_nicolson(u_initial,alpha,Nt,Nx):      
    u_values = np.zeros((Nt, Nx))
    u_values[0, :] = u_initial
    for n in range(1, Nt):
        u_values[n, :] = crank_nicolson_solver(u_values[n-1, :], make_B(Nx), make_identity_matrix(Nx), alpha)
    return u_values
#Getting inverse with LU
def LU_doolittle_inverse(mat, n):
    for i in range(n):
        for j in range(n):
            if i > 0 and i <= j:  # upper triangular matrix
                summation = sum(mat[i][k] * mat[k][j] for k in range(i))
                mat[i][j] -= summation
            if i > j:  # lower triangular matrix
                summation = sum(mat[i][k] * mat[k][j] for k in range(j))
                mat[i][j] = (mat[i][j] - summation) / mat[j][j]
    inverse = [[0] * n for _ in range(n)]
    for col in range(n):
        b = [0] * n
        b[col] = 1
        c = [0] * n
        for i in range(n):
            summation = sum(mat[i][k] * c[k] for k in range(i))
            c[i] = b[i] - summation
        x = [0] * n
        for i in range(n - 1, -1, -1):
            summation = sum(mat[i][k] * x[k] for k in range(i + 1, n))
            x[i] = (c[i] - summation) / mat[i][i]
        for i in range(n):
            inverse[i][col] = x[i]
    return inverse
#Solve with LU decomposition
def forward_substitution_LU(A, b):
    n = len(A)
    y = [0 for _ in range(n)]
    y[0] = b[0][0] 
    for i in range(1, n):
        sum = 0
        for j in range(i):
            #print(f"A[{i+1}][{j+1}]y[{j+1}]",end="+")
            sum += A[i][j] * y[j]
        print("")
        y[i] = (b[i][0] - sum) 

    return y

def backward_substitution(A, y):
    n = len(A)
    x = [0 for _ in range(n)]
    x[n-1] = y[n-1] / A[n-1][n-1]
    for i in range(n-2, -1, -1):
        sum = 0
        for j in range(i+1, n):
            sum += A[i][j] * x[j]
        x[i] = (y[i] - sum) / A[i][i]
    return x

def solve_linear_system_LU(mat, b):
    lu_mat = LU_doolittle(mat, len(mat))
    y = forward_substitution_LU(lu_mat, b)
    x = backward_substitution(lu_mat, y)
    return x
#multivariable newtons method
def newton_method_multivariable(f, Jacobian, x0, tol=1e-2, max_iter=1000):
    x = x0
    for i in range(max_iter):
        jacob=Jacobian(x)
        delta_x = solve(jacob, multiply_1dmatrix_by_constant(f(x),1))
        x = Subtract_matrices(x , delta_x)
        #print(x)
        if norm(delta_x) < tol:
            return x, i +1
    Print("Newton's method did not converge within the maximum number of iterations")
def solve(A,B):#solve Ax=B; x=A^{-1}B
    A_inv=gauss_jordan_inverse(A)
    if isinstance(B[0], (int, float)):#treat B as a (len(B),1) matrix instead of an array
        B = [[elem] for elem in B]
    result=matrix_multiplication(A_inv,B)
    return result
def Subtract_matrices(A, B):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        print("Dimensions don't match")
        return None
    result = [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return result
#calculate jacobian without defining
def subtract_2_1darray(A,B):
    if len(A) != len(B) :
        print("Dimensions don't match")
        return None
    result = [[A[i] - B[i]] for i in range(len(A))]
    return result
def add_1dmatrix(A,B):
    if len(A) != len(B) :
        print("Dimensions don't match")
        return None
    result = [[A[i][0] - B[i][0]] for i in range(len(A))]
    return result
def multiply_1dmatrix_by_constant(array, constant):
    result=[0 for i in range (0,len(array))]  
    for i in range(0,len(array)):
        result[i]=array[i][0]*constant
    return result
def jacobian_using_epsilon(x,func,ep=10**(-2)):
    J = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        x_plus_ep = np.array(x)
        x_plus_ep[i] += ep
        J[:, i] = multiply_1dmatrix_by_constant(subtract_2_1darray(func(x_plus_ep),func(x)),1/ep) 
    return J
#finite element method
def finite_element_method(E, A, L, num_elements):
    element_length = L / num_elements
    nodes = np.linspace(0, L, num_elements + 1)
    elements = np.array([np.arange(i, i + 2) for i in range(num_elements)])
    mat_b = np.zeros((num_elements + 1, num_elements + 1))
    Force = np.zeros(num_elements + 1)
    B = make_B(2) 
    for element in elements:
        x_a, x_b = nodes[element]
        length = x_b - x_a
        ke = (E * A / length) * B
        mat_b[element[:, None], element] += ke
        Force[element] += np.zeros(2)  
    mat_b[0, :] = 0
    mat_b[0, 0] = 2
    Force[0] = 5
    u = np.linalg.solve(mat_b, Force)
    plt.plot(nodes, u, marker='o')
    plt.show()
#multivar secant method
def multivariable_secant_method(g,x0,tol=1e-5,max_iter=100):
    x_prev = x0
    for i in range(max_iter):
        x_new = g(x_prev)
        x_hash = subtract_2_1d_array(x_new, x_prev)
        print(x_hash)
        err = norm1(x_hash)
        if err < tol:
            return x_new, i + 1
        x_prev = x_new
    print("Try different g(x)")
#norm
def norm1(arr):#arr is a 1d array
    sum1=0
    for i in range(0,len(arr)):
        #print(arr)
        sum1=sum1+arr[i]*2
    return sum1**(0.5)

#leap frog and verlet
def semi_implicit_euler(dxdt, dvdt, xo, vo, dt, steps):
    x = np.zeros(steps) # Initialize arrays for positions and velocities
    v = np.zeros(steps)
    x[0] = xo
    v[0] = vo
    for i in range(1, steps):
        v[i] = v[i-1] + dvdt(x[i-1], v[i-1], i*dt) * dt
        x[i] = x[i-1] + dxdt(x[i-1], v[i], i*dt) * dt        
    return x, v

def verlet(dvdt, xo, vo, dt, steps):
    x = [0] * steps
    v = [0] * steps
    x[0] = xo
    x[1] = x[0] + vo * dt + 0.5 * dvdt(x[0]) * dt**2
    v[0] = vo
    # Perform Verlet integration
    for i in range(2, steps):
        x[i] = 2 * x[i-1] - x[i-2] + dvdt(x[i-1]) * dt**2
        v[i] = v[i-1] + 0.5 * (dvdt(x[i]) + dvdt(x[i-1])) * dt

    return x, v

def velocity_verlet(dvdt, xo, vo, dt, steps):
    x = [0] * steps  # Initialize lists with zeros
    v = [0] * steps    
    x[0] = xo
    v[0] = vo    
    for i in range(1, steps):        
        x[i] = x[i-1] + v[i-1] * dt + 0.5*dvdt(x[i-1]) * dt**2
        v[i] = v[i-1] + 0.5*(dvdt(x[i]) + dvdt(x[i-1]))* dt        
    return x, v

def leap_frog(dvdt, xo, vo, dt, steps):
    x = [0] * steps  # Initialize lists with zeros
    v = [0] * steps
    x[0] = xo
    v[0] = vo                   
    for i in range(1, steps):
        v_half = v[i-1] + dvdt(x[i-1]) * (dt/2)
        x[i] = x[i-1] + v_half * dt
        v[i] = v_half + 0.5 * dvdt(x[i]) * dt
    return x,v
#linear equation solver with cholsky
def solve_linear_system_cholsky(mat, b):
    lu_mat = cholesky_decomp(mat)
    y = forward_substitution_cholsky(lu_mat, b)
    x = backward_substitution(transpose_matrix(lu_mat), y)
    return x

def transpose_matrix(matrix):
    rows = len(matrix)
    columns = len(matrix[0])
    result = [[0] * rows for _ in range(columns)]
    for i in range(rows):
        for j in range(columns):
            result[j][i] = matrix[i][j]
    return result
def forward_substitution_cholsky(A, b):
    n = len(A)
    y = [0 for _ in range(n)]
    y[0] = b[0][0] / A[0][0]
    for i in range(1, n):
        sum = 0
        for j in range(i):
            sum += A[i][j] * y[j]
        print("")
        y[i] = (b[i][0] - sum) / A[i][i]
    return y
#gauss jacobi linear equation solver
def GaussJacobi(a,b,ep):#Gauss-Jacobi main code
    n=len(a[0])
    guess=[0.00]*n
    counter=0
    prev=[0]*n
    difference=[0]*n
    while True:
        counter+=1
        iterative_jacobi(a, b, guess)#updates value of guess
        for i in range(len(guess)):
            difference[i]=guess[i]-prev[i]#makes list of differnce
        if max(difference)<ep :#if all differneces less than ep then break
            break        
        else:#else update value of prev
            for r in range(len(guess)):
                prev[r]=guess[r]
    return guess,counter
def iterative_jacobi(a,b,guess):#jacobi 1 iteration
    n=len(a)
    for i in range(0,n):
        temp=b[i][0]
        for j in range(0,n):
            if (i==j):
                pass
            else:
                temp=temp-(a[i][j]*guess[j])
        guess[i]=temp/a[i][i]
    return guess
#gauss seidel
def gauss_seidel(a,b,guess,ep=10**(-6),max_iter=20):
    n=len(a)
    guess_prev=[]    
    for row in range(n): 
        temp=[]
        for column in range(len(a[0])):
            temp.append(0.0)
        guess_prev.append(temp)
    for itera in range(max_iter):    #maximum no fo iterations is 20         
        for k in range(n):
            guess_prev[k][0]=guess[k][0]
        for i in range(n):
            temp=b[i][0]
            for j_k1 in range(i):
                temp=temp-a[i][j_k1]*guess[j_k1][0]  
            for j_k in range(i+1,n):
                temp=temp-a[i][j_k]*guess[j_k][0]
            guess[i][0]=temp/a[i][i]
        diff=0.0
        for iu in range(n):
            diff=diff+abs(guess[iu][0]-guess_prev[iu][0])
        
        if diff<ep :#max of difference list is less than ep
            return guess,itera
            break
#gauss jordan inverse
def gauss_jordan_inverse(matrix):
    n = len(matrix)    
    MatrixAndIdentity = [row + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(matrix)]#create [matrix|I]    
    for i in range(n):
        if MatrixAndIdentity[i][i] == 0:
            MatrixAndIdentity = swap_to_non_zero(MatrixAndIdentity, i)#give i as we dont want to change already reduced elements
            if MatrixAndIdentity is None:
                print("Cant make one of the diagonals of one of the rows non zero.")
                return None        
        MatrixAndIdentity = make_diagonals_one(MatrixAndIdentity, i)
        MatrixAndIdentity = make_rest_of_column_zero(MatrixAndIdentity)    
    inverse_matrix = [row[n:] for row in MatrixAndIdentity]    
    return inverse_matrix
def check_zero(a):
    ctr = -1
    for i in range(len(a)):
        if a[i][i] == 0:
            ctr = i
    return ctr

def swap_to_non_zero(a, n1):
    maxi, ind = 0, -1
    for i in range(len(a)):
        if abs(a[i][n1]) > maxi:
            maxi = abs(a[i][n1])
            ind = i
    if ind != -1:
        a[n1], a[ind] = a[ind], a[n1]
        return a
    else:
        print("All elements in a row are 0, so Gauss-Jordan elimination can't be applied.")
        return None

def make_diagonals_one(a, i):
    temp = a[i][i]
    for k in range(len(a[0])):
        a[i][k] = a[i][k] / temp
    return a

def make_rest_of_column_zero(a):
    for i in range(len(a)):
        for j in range(len(a)):
            if i == j:
                pass
            else:
                p = a[j][i] / a[i][i]
                for k in range(len(a[0])):
                    a[j][k] = round(a[j][k] - (a[i][k]) * p, 5)
    return a
#cholsky decomposition
def cholesky_decomp(mat):
    global result,n
    n=len(mat)
    result = []
    for i in range(n):
       row = []
       for j in range(n):
          row.append(0)
       result.append(row)
    for i in range(n):
        for j in range(i+1):
            s1=0
            if i==j:
                for k in range(0,j):
                    s1+=pow(result[j][k],2)
                result[j][i]=(pow((mat[j][i]-s1),0.5))#considering only positive sign                
            else:
                for k in range(i+1):
                    s1+=(result[i][k]*result[j][k])
                if result[j][j]>0:
                    result[i][j]=(mat[i][j]-s1)/result[j][j]
    return result

#solution using gauss jordan
def solve_linear_system_gauss_jordan(coeff_matrix, constant):
    augmented_matrix = [row + [constant[i]] for i, row in enumerate(coeff_matrix)]#[coeff_matrix | constant]
    # Gauss-Jordan RREF
    n = len(augmented_matrix)
    for i in range(n):
        if augmented_matrix[i][i] == 0:
            augmented_matrix = swap_to_non_zero(augmented_matrix, i)
            if augmented_matrix is None:
                print("Unable to solve the system, no unique solution.")
                return None
        augmented_matrix = make_diagonals_one(augmented_matrix, i)
        augmented_matrix = make_rest_of_column_zero(augmented_matrix)
    solution = [row[-1] for row in augmented_matrix]#last column as it is [coeff_matrix | constant]
    return solution
#LU decomposition(l1=l2=l3=1)
def LU_doolittle(mat, n):
    for i in range(n):
        for j in range(n):
            if i > 0 and i <= j:  # upper triangular matrix
                summation = sum(mat[i][k] * mat[k][j] for k in range(i))
                mat[i][j] -= summation
            if i > j:  # lower triangular matrix
                summation = sum(mat[i][k] * mat[k][j] for k in range(j))
                mat[i][j] = (mat[i][j] - summation) / mat[j][j]
    return mat



#copy matrix
def copy_matrix(mat1):#to prevent elemnts of two matrix from not pointing to the same adress
    copym=[[0 for _ in range (0,len(mat1))] for _ in range(0,len(mat1[0]))]
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            copym[i][j]=mat1[i][j]
    return copym
#Poly root; from previos course get the roots of a polynomial
import cmath#for sqrt as compliex number may be involved
#roots = PolynomialRoots(coefficients).roots; then round based on tolerance default(10^(-6))
def solvepoly(a, x):
    n = len(a)-1
    p =0
    for i in range(len(a)):
        p+=a[i]*pow(x,n-i) #(coeff of x^n)*x^(n)
    return p
class PolynomialRoots:
    def __init__(self, coefficients, tol=1e-6):
        self.coefficients = coefficients
        self.roots = self.polyRoots(tol)
    def laguerre(self, x, ep):
        n = len(self.coefficients) - 1
        for i in range(15):#setting maximum no of iterations as 15
            p,d,dd = self.solvepoly(x),self.deriv(x),self.doublederiv(x)
            if abs(p) < ep: # since x solves for a
                return x
            g = d/p
            h = g*g - dd/p
            f = cmath.sqrt((n - 1)*(n*h - g*g))#can return a complex value as nh < g^2,is possible
            if abs(g+f) > abs(g-f): #finds greateest denominator
                dx = n/(g+f)
            elif abs(g-f) >= abs(g+f): #if both denoms are equal then either can be chosen
                dx = n/(g-f)
            x = x - dx
            if abs(dx) < ep:#if error is less than ep
                return x
        print('Too many iterations') #after 15 iteration
    def solvepoly(self, x):
        n = len(self.coefficients)-1
        p =0
        for i in range(len(self.coefficients)):
            p+=self.coefficients[i]*pow(x,n-i) #(coeff of x^n)*x^(n)
        return p
    def deriv(self, x):
        n=len(self.coefficients)-1
        d=0.0
        for i in range(len(self.coefficients)-1):
            d+=(self.coefficients[i]*pow(x,n-i-1)*(n-i)) #(coeff of x^n)*n*x^(n-1)
        return d
    def doublederiv(self, x):
        n=len(self.coefficients)-1
        dd=0.0
        for i in range(len(self.coefficients)-2):
            dd+=(self.coefficients[i]*pow(x,n-i-2)*(n-i-1)*(n-i)) #(coeff of x^n)*n*(n-1)*x^(n-2)
        return dd
    def deflate(self, root):
        a=self.coefficients
        temp=[]
        n=len(self.coefficients)
        for j in range(n):
            temp.append(0)    #makes a list of n lenght
        for i in range(n-1):
            temp[i]=a[i]
            self.coefficients[i+1]=self.coefficients[i+1]+root*self.coefficients[i]#since a is a local variable it wont be changes in the main code
        temp.pop(n-1) #removes 0 at the end of the list
        return temp
    def polyRoots(self, tol):
        roots = []
        n = len(self.coefficients) - 1
        x = 1.5  # starting value
        for i in range(n):
            x = self.laguerre(x, tol)
            if abs(x.imag) < tol:
                x = x.real
            roots.append(x)
            self.coefficients = self.deflate(x)
        return roots


#some matrix algebra
def make_B(n):
    B = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                B[i][j] = 2
            elif i == (j - 1) or i == (j + 1):
                B[i][j] = -1
            else:
                B[i][j] = 0
    return B

def make_identity_matrix(n):
    B = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                B[i][j] = 1
            else:
                B[i][j] = 0
    return B
def matrix_mul_B_is_1D(A,B):
    if len(A[0]) != len(B):
        print("Columns in matrix 1 and rows in matrix 2 dont match")
        return None
    if isinstance(B[0], (int, float)):#treat B as a (len(B),1) matrix instead of an array
        B = [[elem] for elem in B]
    result = [[0] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][0] += A[i][k] * B[k][j]
    result = [elem for sublist in result for elem in sublist]# Flatten the result matrix to a 1D array
    return result
def multiply_matrix_by_constant(matrix, constant):
    result = [[element * constant for element in row] for row in matrix]
    return result
def Add_matrices(A, B):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        print("Dimensions Don't match")
        return None
    result = [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return result
def Subtract_matrices(A, B):
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        print("Dimensions don't match")
        return None
    result = [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
    return result
def matrix_multiplication(m1,m2):
    row_m1,column_m1=len(m1),len(m1[0])
    row_m2,column_m2=len(m2),len(m2[0])
    result = []
    for i in range(row_m1):
       row = []
       for j in range(column_m2):
          row.append(0)
       result.append(row)
    #print(result)
    for i in range(row_m1):
       for j in range(column_m2):
          for k in range(column_m1):
             result[i][j] += m1[i][k] * m2[k][j]
    return result

#plot function f(x) between a and b
def plot_function(func, a=0, b=100):
    x_values = np.linspace(a, b, 100)
    y_values = func(x_values)
    plt.plot(x_values, y_values, label='f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()
#round to n
def round_to_n(lis1,n=6):#rounds all elements in a list to n decimal places
    for i in range (0,len(lis1)):
        lis1[i]=round(lis1[i],n)
    return lis1
#misc functions just for convinience
def print_matrix(matrix):
    for row in matrix:
        print(row)
#norm
def norm(arr):#arr is a 1d array
    sum1=0
    for i in range(0,len(arr)):
        sum1=sum1+arr[i][0]**2
    return sum1**(0.5)

#lorentz equation
def rk4_lorentz(lorentz_equations,t0, x0, y0, z0, tn, h, sigma, rho, beta):
    n = round((tn - t0) / h)
    u = np.array([x0, y0, z0])
    tlis = [t0]
    xlis = [x0]
    ylis = [y0]
    zlis = [z0]

    for i in range(1, n + 1):
        k1 = h * lorentz_equations(t0, u, sigma, rho, beta)
        k2 = h * lorentz_equations(t0 + h / 2, u + k1 / 2, sigma, rho, beta)
        k3 = h * lorentz_equations(t0 + h / 2, u + k2 / 2, sigma, rho, beta)
        k4 = h * lorentz_equations(t0 + h, u + k3, sigma, rho, beta)

        u = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t0 = t0 + h

        tlis.append(t0)
        xlis.append(u[0])
        ylis.append(u[1])
        zlis.append(u[2])

    return np.array(tlis), np.array(xlis), np.array(ylis), np.array(zlis)
