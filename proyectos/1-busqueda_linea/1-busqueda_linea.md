---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.6
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
from scipy import linalg
from scipy.optimize import rosen
from scipy.linalg import LinAlgError
import numdifftools as nd
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from pprint import pprint
plt.rcParams['figure.figsize'] = (20,10)
```

# Búsqueda de línea

## Condición de Wolfe (W1)

```python
def w1(f, x, gx, p, alpha, c1):
    """
    Revisa si la condición de Armijo
    
        f(x+alpha*p) <= f(x) + alpha*c1*(gx'p)
    
    se satisface.
    """
    return f(x+alpha*p) <= f(x) + alpha*c1*np.dot(gx, p)
```

## Encontrar la dirección de descenso $p$

```python
def encontrar_direccion_descenso(gx, f=None, x=None, direction='max', **kwargs):
    """ Encuentra la dirección de descenso de f en el punto x con el método indicado.
    
    Parámetros
    ----------
    method : string
        Uno de 'newton', 'max', o 'descoor'. El último da sólo la coordenada de mayor descenso.
        
    Regresa
    -------
        ndarray: La dirección de descenso.    
    """
    if direction == 'descoor':
        return descoor(gx)
    elif direction == 'max':
        return -gx
    elif direction == 'newton':
        return newton_dir(f, x, gx)
    else:
        raise ValueError('Dirección inválida')
```

```python
def descoor(gx):
    """
    Regresa
    -------
    ndarray
        La coordenada de mayor descenso de f en x.
    """ 
    n = gx.shape[0]
    k = np.argmax(gx)
    
    if gx[k] > 0:
        return -np.identity(n)[k]
    else:
        return np.identity(n)[k]      
```

```python
def newton_dir(f, x, gx):
    """ 
    Regresa
    -------
    ndarray
        La dirección de Newton de f en x.
    """
    H = nd.Hessian(f)(x)
    
    try: 
        return linalg.cho_solve(linalg.cho_factor(H), -gx)
    except LinAlgError:
        return linalg.solve(H,-gx)
```

## Tamaño del paso: $\alpha$

```python
def encontrar_tamano_paso(f, x, gx, p, c1=1e-4, alpha_iter=20, alpha=1, alpha_method='hibrido', **kwargs):
    """Aproxima el tamaño óptimo del paso a dar.
    
    Parameters
    ----------
    
    Returns
    -------
    double
        Tamaño de paso que satisface (W1).
    
    """
    if alpha_method == 'backtracking':
        return backtracking(f, x, gx, p, c1=c1, alpha_iter=alpha_iter, alpha=1)
    elif alpha_method == 'interpol':
        return interpolacion_cuadratica(f, x, gx, p, c1=c1, alpha_iter=alpha_iter, alpha=1)
    elif alpha_method == 'hibrido':
        return hibrido(f, x, gx, p, c1=c1, alpha_iter=alpha_iter, alpha=1)
    else:
        raise ValueError('Método inválido')
```

```python
def backtracking(f, x, gx, p, c1=1e-4, alpha_iter=20, alpha=1):
    """Encuentra por backtracking la tasa de aprendizaje en una iteración.
    """
    
    n_iter = 0
    
    while (not w1(f, x, gx, p, alpha, c1)) and (n_iter < alpha_iter):
        alpha /= 2
        n_iter += 1
        
    return alpha
```

```python
def interpolacion_cuadratica(f, x, gx, p, c1=1e-4, alpha_iter=20, alpha=1):
    """ Elección del tamaño de paso en búsqueda de línea a través de minimizar un polinomio cuadrático.
    """
    g = lambda t: f(x+t*p)
    g0 = g(0)
    gal = g(alpha)
    dg0 = np.dot(gx, p)
    
    num_iter = 0
    while (not w1(f, x, gx, p, alpha, c1)) and (num_iter < alpha_iter):
        alpha = -dg0*alpha**2/(2*(gal-g0-dg0*alpha)) 
        gal = g(alpha)
        num_iter += 1
        
    return alpha
```

```python
def hibrido(f, x, gx, p, c1=1e-4, alpha_iter=20, alpha=1, tol_alpha=1e-3):
    """ Método híbrido para encontrar el tamaño de paso en búsqueda de línea.
    """
    
    bt_flag = inter_flag = False
    alpha_bt = alpha
    alpha_inter = alpha
    
    num_iter = 0
    
    if w1(f, x, gx, p, alpha, c1):
        return alpha
    
    while (bt_flag + inter_flag == 0) and num_iter < alpha_iter:
        alpha_bt = alpha_bt/2
        bt_flag = w1(f, x, gx, p, alpha_bt, c1)
        
        alpha_inter = interpolacion_cuadratica(f, x, gx, p, c1=c1, alpha_iter=1, alpha=alpha_bt)
        inter_flag = w1(f, x, gx, p, alpha_inter, c1)
        num_iter += 1

    if num_iter == alpha_iter:
        return 1e-2
    elif linalg.norm(alpha*p) < tol_alpha:
        return 1e-2
    elif inter_flag:
        if bt_flag:
            alpha = max(alpha_inter, alpha_bt)
            return alpha
        else:
            return alpha_inter
    elif bt_flag:
        return alpha_bt
```

## Búsqueda de línea

```python
def busqueda_linea(f, x0, tol_g=1e-8, tol_p=1e-5, max_iter=250, **kwargs):
    """ Método de descenso por búsqueda de línea para minimizar una función 
        f:R^n --> R de clase C2.
    
    Parameters
    ----------
    f : función
        La función objetivo.
    x0 : ndarray
        Estimación inicial.
    tol : double
        Tolerancia para la norma del gradiente.
    max_iter : int
        Número máximo de iteraciones.
        
    Returns
    -------
    double
        Mínimo local de f.
    int
        Número de iteraciones que se realizaron.
    ndmatrix
        Valores que tomó x durante la optimización.
    """
    
    W = np.copy(x0)
    x = np.copy(x0)
    n_iter = 0
    
    grad = nd.Gradient(f)
    gx = grad(x)
    
    while (linalg.norm(gx) > tol_g) and (n_iter < max_iter):
      
        if (n_iter > 0):
            if (linalg.norm(x-W[n_iter-1,:])<linalg.norm(W[n_iter-1,:]) *tol_p) or (linalg.norm(p)<tol_p):
                break
        
        p = encontrar_direccion_descenso(gx, f=f, x=x, **kwargs)
        
        alpha = encontrar_tamano_paso(f, x, gx, p, **kwargs)
              
        x = x + alpha*p
        gx = grad(x)
        
        W = np.vstack([W,x])
        n_iter +=1
    
    return ({'x*':x, 'n_iter':n_iter, 'z*':f(x), 'gx':gx}, W)    
```

## Funciones de prueba

```python
def rastrigin(x):
    n = x.shape[0]
    return 10*n + np.sum(x**2-10*np.cos(2*np.pi*x))
```

```python
def griewank(x):
    n = x.shape[0]
    return 1/4000*np.sum(x**2) - np.prod(np.cos(x/np.sqrt(np.arange(n)+1)))+1
```

```python
def ackley(x, a=20, b=0.2, c=2*np.pi):
    n = x.shape[0]
    return -a*np.exp(-b*np.sqrt(1/n*np.sum(x**2))) -np.exp(1/n*np.sum(np.cos(c*x)))+a+np.exp(1)
```

```python
def branin(x, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    return a*(x[1]-b*x[0]**2+c*x[0]-r)**2 + s*(1-t)*np.cos(x[0])+s  
```

```python
def easom(x):
    return -np.cos(x[0])*np.cos(x[1])*np.exp(-(x[0]-np.pi)**2-(x[1]-np.pi)**2)
```

```python
def second_schaffer(x):
    return 0.5 + (np.sin(x[0]**2-x[1]**2)**2-0.5)/((1+0.001*np.sum(x**2))**2)
```

```python
def shubert(x):
    i = np.arange(5)+1
    return np.sum(i*np.cos((i+1)*x[0]+i))*np.sum(i*np.cos((i+1)*x[1]+i))
```

## Pruebas y figuras

```python
def plot_path(xmin, xmax, ymin, ymax, f, opt, pathn, pathm):
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    x, y = np.meshgrid(x, y)
    z = f(x,y)
    
    
    cs = plt.contour(x, y, z, 30, alpha=0.5)
    plt.clabel(cs, inline=1, fontsize=10)
    plt.grid(True)
    plt.plot(opt[0], opt[1], 'r*', markersize=25)
    
    for i in range(pathn.shape[0]-1):
        x0 = pathn[i,0]
        y0 = pathn[i,1]
        dx = pathn[i+1,0] - x0
        dy = pathn[i+1,1] - y0
        plt.arrow(x0, y0, dx, dy, color='orange', head_width=0.05)
    for i in range(pathm.shape[0]-1):
        x0 = pathm[i,0]
        y0 = pathm[i,1]
        dx = pathm[i+1,0] - x0
        dy = pathm[i+1,1] - y0
        plt.arrow(x0, y0, dx, dy,head_width=0.05)
```

### Rosenbrock

```python
resn, pathn = busqueda_linea(rosen, np.array([2,2]), direction='newton')
resm, pathm = busqueda_linea(rosen, np.array([2,2]), direction='max')
plot_path(0, 3, -2, 5, lambda x,y: 100*(y-x**2)**2+(x-1)**2, [1,1], pathn, pathm)
```

```python
pprint(resn)
print()
print('---------------')
print()
pprint(resm)
```

Converge bien en Newton (16 iteraciones) pero no en máximo descenso:


### Rastrigin

```python
resn, pathn = busqueda_linea(rastrigin, np.array([0.4, 0.3]), direction='newton')
resm, pathm = busqueda_linea(rastrigin, np.array([0.4, 0.3]), direction='max')
plot_path(-4, 4, -4, 4, lambda x,y: 20+x**2+y**2-10*np.cos(2*np.pi*x)-10*np.cos(2*np.pi*y), [0,0], pathn, pathm)
```

```python
pprint(resn)
print()
print('---------------')
print()
pprint(resm)
```

Se va a mínimos locales en ambos casos, pero igual es mejor el de máximo descenso.


### Griewank

```python
resn, pathn = busqueda_linea(griewank, np.array([0.4, 0.3]), direction='newton')
resm, pathm = busqueda_linea(griewank, np.array([0.4, 0.3]), direction='max')
plot_path(-4, 4, -1, 1, lambda x,y: (x**2+y**2)/4000-np.cos(x)-np.cos(y/np.sqrt(2))+1, [0,0], pathn, pathm)
```

```python
pprint(resn)
print()
print('---------------')
print()
pprint(resm)
```

En newton converge y en máximo converge un poquito más lento.


### Ackley

```python
resm, pathm = busqueda_linea(ackley, np.array([0, 1.5]), direction='max')
resn, pathn = busqueda_linea(ackley, np.array([0, 1.5]), direction='newton')
ack = lambda x,y: -20*np.exp(-0.2/2*(x**2+y**2))-np.exp(1/2*(np.cos(2*np.pi*x))+np.cos(2*np.pi*y))+20+np.exp(1)
plot_path(-1, 1, -2, 2, ack, [0,0], pathn, pathm)
```

```python
pprint(resn)
print()
print('---------------')
print()
pprint(resm)
```

Converge en máximo descenso (44 iteraciones) pero no en Newton.


### Branin

```python
resn, pathn = busqueda_linea(branin, np.array([-4, 13]), direction='newton')
resm, pathm = busqueda_linea(branin, np.array([-4, 13]), direction='max')
bran = lambda x,y: (y-(5.1/(4*np.pi**2))*x**2+(5/np.pi)*x-6)**2 + 10*(1-(1/(8*np.pi)))*np.cos(x)+10
plot_path(-5, 5, 10, 15, bran, [-np.pi,12.275], pathn, pathm) 
```

```python
pprint(resn)
print()
print('---------------')
print()
pprint(resm)
```

Converge en 37 iteraciones con máximo descenso y en 4 con Newton.


### Easom

```python
resn, pathn = busqueda_linea(easom, np.array([5, 5]), direction='newton')
resm, pathm = busqueda_linea(easom, np.array([5, 5]), direction='max')
eas = lambda x,y: -np.cos(x)*np.cos(y)*np.exp(-(x-np.pi)**2-(y-np.pi)**2)
plot_path(2, 6, 2, 6, eas, [np.pi,np.pi], pathn, pathm) 
```

```python
pprint(resn)
print()
print('---------------')
print()
pprint(resm)
```

En ambos casos se atora y casi no avanza.


### Schaffer (2)

```python
resn, pathn = busqueda_linea(second_schaffer, np.array([1, 1]), direction='newton')
resm, pathm = busqueda_linea(second_schaffer, np.array([1, 1]), direction='max')
sha2 = lambda x,y: 0.5 + (np.sin(x**2-y**2)**2-0.5)/(1+0.001*(x**2+y**2))**2
plot_path(-2, 2, -2, 2, sha2, [0,0], pathn, pathm) 
```

```python
pprint(resn)
print()
print('---------------')
print()
pprint(resm)
```

Newton converge rápido (en 2 iteraciones) y máximo no converge.


### Shubert

```python
resn, pathn = busqueda_linea(shubert, np.array([3, 1]), direction='newton')
resm, pathm = busqueda_linea(shubert, np.array([3, 1]), direction='max')
def shub(x,y):
    resx = resy = 0
    for i in range(1,6):
        resx += i*np.cos((i+1)*x+i)
        resy += i*np.cos((i+1)*y+i)
    return resx*resy
plot_path(0, 45, 0, 6, shub, [36.27,5.48], pathn, pathm) 
```

```python
pprint(resn)
print()
print('---------------')
print()
pprint(resm)
```

Newton se atora en un mínimo local después de tres iteraciones y máximo descenso converge en 15.


## Compación con los métodos no-híbridos que vimos

```python
def comparar(f, x, direction):
    x = np.array(x)
    for m in ['hibrido', 'backtracking', 'interpol']:
        resn, _ = busqueda_linea(f, x, direction=direction, alpha_method=m)
        print(f"Método {m}")
        print(f"    {resn['n_iter']} iteraciones")
        print(f"    La norma del gradiente es {linalg.norm(resn['gx'])}")
```

### Rosenbrock

```python
comparar(rosen, [2,2], 'newton')
```

### Rastrigin

```python
comparar(rastrigin, [0.4, 0.3], 'max')
```

### Griewank

```python
comparar(griewank, [2,0], 'newton')
```

### Ackley

```python
comparar(ackley, [0, 1.5], 'newton')
```

### Branin

```python
comparar(branin, [-4, 13], 'newton')
```

### Easom

```python
comparar(easom, [5,5], 'max')
```

### Segunda función de Schaffer

```python
comparar(second_schaffer, [1,1], 'newton')
```

### Shubert

```python
comparar(shubert, [3,1], 'max')
```
