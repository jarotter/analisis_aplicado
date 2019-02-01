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

# Laboratorio 1 - métodos de descenso

```python
# Parche para sibling imports
sys.path.append('..')
```

```python
# Imports genéricos
import numpy as np
import sys
from utils import gradiente
```

```python
def encontrar_alfa(f, x, grad, p, c1=0.1, max_inner_iter=10):
    """Encuentra por búsqueda binaria la tasa de aprendizaje en una iteración.
    
    Entradas:
    ---------
    f : función
        El objetivo que se está optimizando.
    x : ndarray
        Punto actual.
    grad : ndarray
        Gradiente de f en x.
    p : ndarray
        Dirección de descenso.
    c1 : double
        Constante de wolfe.
    max_iner_iter : int
        Cota para el número de iteraciones o equivalentemente, 
        cota inferior para alfa de la forma 2^{-max_inner_iter}<=alfa
        
    Regresa:
    --------
    alpha : double
        Una buena tasa de aprendizaje.
    """
    
    n_iter = 0
    alpha = 1
    
    while f(x+alpha*p) > f(x)+alpha*c1*np.dot(grad,p) & n_iter < max_inner_iter:
        alfa /= 2
        n_iter += 1
        
    return alpha
```

```python
def descoor(grad):
    """ Elige p por la coordenada de mayor descenso.
    """
    k = np.argmax(grad)
    if grad > 0:
        return -np.identity(n)[k]
    else:
        return np.identity(n)[k]      
```

```python
def max_desc(f, x):
    """ Elige p de máximo descenso.
    """
    return -gradiente(f,x)
```

```python
def
```

```python
def busqueda_linea(f, x, tol=1e-5, max_iter=300, method='newton', **kwargs):
    """ Método de descenso por coordenadas para minimizar una función f:R^n --> R de clase C2.
    
    Entradas:
    ---------
    f : función
        La función objetivo.
    x0 : ndarray
        Valor inicial.
    tol : double
        Tolerancia.
    max_iter : int
        Número máximo de iteraciones.
    method : string
        Uno de 'newton', 'descoor', o 'max' para elegir la dirección de descenso.
        
    Regresa:
    --------
    x* : double
        Mínimo local de f.
    n_iter : int
        Número de iteraciones que se realizaron.
    W : 
        Valores que tomó x durante la optimización.
    """
    n = x.shape[0]
    W = np.copy(x)
    
    n_iter = 0
    grad = gradiente(f,x)
    while np.linalg.norm(grad) > tol and n_iter < max_iter:
        
        if method == 'descoor':
            p = descoor(grad)
        elif method == 'max':
            p = max_descent(f,x):
        elif method == 'newton':
            p = newton_dir(f,x)
        else:
            raise ValueError('Método inválido')
            
        alfa = encontrar_alfa(f,x,grad,p,**kwargs)
        x = x + alfa*p
        grad = gradiente(f,x)
        
        W = np.vstack([W,x])
        
    return (x, n_iter, W)
    
```

```python
def inner(x=1000):
    return f'inner function with x={x} and y={y}'

def outer(y=10, **kwargs):
    print(f'outer function with outerx={y}')
    print(inner(**kwargs))
    
```
