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
import numpy as np
import numdifftools as nd

from scipy.optimize import rosen
```

# Laboratorio 1 - métodos de descenso

Método general de búsqueda de línea para descenso por tres direcciones: Newton, máximo descenso y coordenada de máximo descenso.

En este notebook suponemos

$$
\mathcal{C}^2 \ni f: \mathbb{R}^n \to \mathbb{R}
$$

## Método general

```python
def busqueda_linea(f, x, tol=1e-5, max_iter=300, direction='newton', alpha_method='backtracking', **kwargs):
    """ Método de descenso por coordenadas para minimizar una función 
        f:R^n --> R de clase C2.
    
    Entradas
    --------
    f : función
        La función objetivo.
    x : ndarray
        Estimación inicial.
    tol : double
        Tolerancia para la norma del gradiente.
    max_iter : int
        Número máximo de iteraciones.
    direction : string
        Uno de 'newton', 'descoor', o 'max' para elegir la dirección de descenso.
    alpha_method : string
        Uno de 'backtracking' o 'interpool' para elegir el tamaño de paso.
        
    Regresa
    -------
    double
        Mínimo local de f.
    int
        Número de iteraciones que se realizaron.
    ndmatrix
        Valores que tomó x durante la optimización.
    """
    
    n = x.shape[0]
    W = np.copy(x)
    
    n_iter = 0
    grad = nd.Gradient(f)
    gx = grad(x)
    
    while (np.linalg.norm(gx) > tol) and (n_iter < max_iter):
        
        p = encontrar_direccion_descenso(f, x, grad=gx, method=direction)
        
        alfa = encontrar_tamano_paso(f, x, p, gx, alpha_method, **kwargs)
        
        x = x + alfa*p
        gx = grad(x)
        
        W = np.vstack([W,x])
        n_iter +=1
        
    return ({'x*':x, 'n_iter':n_iter, 'W':W})
    
```

## Encontrar $p$

```python
def encontrar_direccion_descenso(f, x, grad=None, method='newton'):
    """ Encuentra la dirección de descenso de f en el punto x con el método indicado.
    
    Parámetros
    ----------
    method : string
        Uno de 'newton', 'max', o 'descoor'. El último da sólo la coordenada de mayor descenso.
        
    Regresa
    -------
        ndarray: La dirección de descenso.    
    """
    if method == 'descoor':
        return descoor(grad)
    elif method == 'max':
        return -grad
    elif method == 'newton':
        return newton_dir(f, x)
    else:
        raise ValueError('Dirección inválida')
```

```python
def descoor(grad):
    """
    Regresa
    -------
    ndarray
        La coordenada de mayor descenso de f en x.
    """ 
    n = grad.shape[0]
    k = np.argmax(grad)
    if grad[k] > 0:
        return -np.identity(n)[k]
    else:
        return np.identity(n)[k]      
```

```python
def newton_dir(f, x):
    """ 
    Regresa
    -------
    ndarray
        La dirección de Newton de f en x.
    """
    H = nd.Hessian(f)(x)
    return np.linalg.solve(H,-nd.Gradient(f)(x))
```

## Encontrar $\alpha$

```python
def encontrar_tamano_paso(f, x, p, grad, c1=10e-4, alpha_iter=20, alpha=1):
    """Aproxima el tamaño óptimo del paso a dar.
        
Entradas
    --------
    c1 : double
        Constante de Armijo o de (W1).
    alpha_iter : int
        Cota para el número de iteraciones o equivalentemente, 
        cota inferior para alfa de la forma 2^{-max_inner_iter}<=alfa
    alpha : double
        La estimación inicial.
        
    Regresa
    -------
    double 
        Una buena tasa de aprendizaje.
    
    """
    if alpha_method == 'backtracking':
        return backtracking(f, x, p, gx, **kwargs)
    elif alpha_method == 'interpol':
        return interpolacion_cuadratica(f, x, p, gx, **kwargs)
    else:
        raise ValueError('Método inválido')
```

```python
def backtracking(f, x, p, grad, c1=10e-4, alpha_iter=20, alpha=1):
    """Encuentra por backtracking la tasa de aprendizaje en una iteración.
    """
    
    n_iter = 0
    
    while (f(x+alpha*p) > f(x)+alpha*c1*np.dot(grad,p)) and (n_iter < alpha_iter):
        alpha /= 2
        n_iter += 1
        
    return alpha
```

```python
def interpolacion_cuadratica(f, x, p, gx, c1=10e-4, alpha_iter=20, alpha=1):
    """ Elección del tamaño de paso en búsqueda de línea a través de minimizar un polinomio cuadrático.
    """
    g = lambda t: f(x+t*p)
    gal = g(alpha)
    g0 = g(0)
    dg0 = np.dot(gx, p)
    
    num_iter = 0
    while (gal>g0+alpha*(c1*dg0)) and num_iter < alpha_iter:
        t = -dg0/(2*(gal-g0-dg0))
        gal = g(alpha)
        num_iter += 1
        
    return alpha
```

## Pruebas

### Función de Rosenbrok

```python
for m in ['descoor', 'max', 'newton']:
    print(f'Ahora trabajando en el método {m}')
    resp = busqueda_linea(rosen, np.array([2,3]), direction=m, c1=10e-4)
    n = resp['n_iter']
    x = resp['x*']
    print(f"tomó {n} iteraciones llegar a {x} con f(x)={rosen(x)}")
          
```

### Función cuadrática

```python
def funcuad(x):
    A = np.array([[1,1,1,1], [1,2,3,4], [1,3,6,10], [1,4,10,20]])
    b = -np.ones(4)
    c = 1
    
    return 1/2*np.dot(np.dot(x,A), x)+np.dot(x,b)+c
```

```python
for m in ['descoor', 'max', 'newton']:
    print(f'Ahora trabajando en el método {m}')
    resp = busqueda_linea(funcuad, np.array([5,5,5,5]), direction=m, alpha_method='interpol', c1=10e-4)
    n = resp['n_iter']
    x = resp['x*']
    print(f"tomó {n} iteraciones llegar a {x} con f(x)={funcuad(x)}")
    print()
          
```

```python

```
