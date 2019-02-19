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
def encontrar_tamano_paso(f, x, gx, p, c1=1e-4, alpha_iter=20, alpha=1, alpha_method='hibrido',
                              hybridization=None, **kwargs):
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
        return hibrido(f, x, gx, p, c1=c1, alpha_iter=alpha_iter, alpha=1, hybridization=hybridization)
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
def which_alpha(alpha_bt, alpha_inter, hybridization):
    """Define cómo continuar el método híbrido depara seleccionar el tamaño dep paso.
    """
    
    if hybridization is None:
        return (alpha_bt, alpha_inter)
    elif hybridization == 'max':
        alpha = max(alpha_bt, alpha_inter)
    elif hybridization == 'min':
        alpha = min(alpha_bt, alpha_inter)
    elif hybridization == 'bt':
        alpha = alpha_bt
    elif hybridization == 'inter':
        alpha = alpha_inter
    else:
        raise ValueError('Hibridación inválida')
    
    return (alpha, alpha)
```

```python
def hibrido(f, x, gx, p, c1=1e-4, alpha_iter=20, alpha=1, hybridization=None, tol_alpha=10e-2):
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
        
        alpha_inter = interpolacion_cuadratica(f, x, gx, p, c1=c1, alpha_iter=1, alpha=alpha_inter)
        inter_flag = w1(f, x, gx, p, alpha_inter, c1)
        
        alpha_bt, alpha_inter = which_alpha(alpha_bt, alpha_inter, hybridization)
        num_iter += 1

    if num_iter > alpha_iter:
        return tol_alpha
    elif inter_flag:
        if bt_flag:
            alpha = max(bt_flag, inter_flag)
            if linalg.norm(alpha*p) <= tol_alpha:
                return alpha
            else:
                return tol_alpha
        else:
            return alpha_inter
    else:
        return alpha_bt
```

## Búsqueda de línea

```python
def busqueda_linea(f, x0, tol_g=1e-8, tol_p=1e-4, max_iter=250, **kwargs):
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
    
    n = x0.shape[0]
    W = np.copy(x0)
    x = np.copy(x0)
    n_iter = 0
    
    grad = nd.Gradient(f)
    gx = grad(x)
    
    while (linalg.norm(gx) > tol_g) and (n_iter < max_iter):
        
        p = encontrar_direccion_descenso(gx, f=f, x=x, **kwargs)
        
        if linalg.norm(p) < tol_p:
            break
        
        alpha = encontrar_tamano_paso(f, x, gx, p, **kwargs)
              
        x = x + alpha*p
        gx = grad(x)
        
        W = np.vstack([W,x])
        n_iter +=1
        
    return ({'x*':x, 'n_iter':n_iter, 'z*':f(x), 'gx':gx})
    
```

## Funciones de prueba

```python
@jit(nopython=True)
def rastrigin(x):
    n = x.shape[0]
    return 10*n + np.sum(x**2-10*np.cos(2*np.pi*x))
```

```python
@jit(nopython=True)
def griewank(x):
    n = x.shape[0]
    return 1/4000*np.sum(x**2) - np.prod(np.cos(x/np.sqrt(np.arange(n)+1)))+1
```

```python
@jit(nopython=True)
def ackley(x, a=20, b=0.2, c=2*np.pi):
    n = x.shape[0]
    return -a*np.exp(-b*np.sqrt(1/n*np.sum(x**2))) -np.exp(1/n*np.sum(np.cos(c*x)))+a+np.exp(1)
```

```python
@jit(nopython=True)
def branin(x, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    return -a*(x[1]-b*x[0]**2+c*x[0]-r)**2 + s*(1-t)*np.cos(x[0])+s  
```

```python
@jit(nopython=True)
def easom(x):
    return -np.cos(x[0])*np.cos(x[1])*np.exp(-(x[0]-np.pi)**2-(x[1]-np.pi)**2)
```

```python
@jit(nopython=True)
def second_schaffer(x):
    return 0.5 + (np.sin(x[0]**2-x[1]**2)**2-0.5)/((1+0.001*np.sum(x**2))**2)
```

```python
@jit(nopython=True)
def shubert(x):
    i = np.arange(5)+1
    return np.sum(i*np.cos((i+1)*x[0]+i))*np.sum(i*np.cos((i+1)*x[1]+i))
```

## Pruebas

```python
busqueda_linea(rosen, np.array([2,2]), direction='newton')
```

```python
busqueda_linea(rastrigin, np.array([0.4, 0.3]), direction='max')
```

```python
busqueda_linea(griewank, np.array([2, 0]), direction='newton')
```

```python
busqueda_linea(ackley, np.array([0, 1.5]), direction='max')
```

```python
busqueda_linea(branin, np.array([-4, 13]), direction='newton')
```

```python
busqueda_linea(easom, np.array([5, 5]), direction='max')
```

```python
busqueda_linea(second_schaffer, np.array([1, 1]), direction='newton')
```

```python
busqueda_linea(shubert, np.array([3, 1]), direction='newton')
```
