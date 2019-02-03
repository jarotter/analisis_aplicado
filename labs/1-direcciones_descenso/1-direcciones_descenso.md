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
def busqueda_linea(f, x, tol=1e-5, max_iter=300, direction='newton', **kwargs):
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
    direction : string
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
    grad = nd.Gradient(f)
    while (np.linalg.norm(grad(x)) > tol) and (n_iter < max_iter):
        
        if direction == 'descoor':
            p = descoor(grad(x))
        elif direction == 'max':
            p = -grad(x)
        elif direction == 'newton':
            p = newton_dir(f,x)
        else:
            raise ValueError('Método inválido')
            
        alfa = encontrar_alfa(f,x,grad(x),p,**kwargs)
        x = x + alfa*p
        
        W = np.vstack([W,x])
        n_iter +=1
        
    return ({'x*':x, 'n_iter':n_iter, 'W':W})
    
```

## Encontrar $\alpha$

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
    
    while (f(x+alpha*p) > f(x)+alpha*c1*np.dot(grad,p)) and (n_iter < max_inner_iter):
        alpha /= 2
        n_iter += 1
        
    return alpha
```

## Encontrar $p$

```python
def descoor(grad):
    """ Elige p por la coordenada de mayor descenso.
    """
    
    n=grad.shape[0]
    k = np.argmax(grad)
    if grad[k] > 0:
        return -np.identity(n)[k]
    else:
        return np.identity(n)[k]      
```

```python
def max_desc(f, x):
    """ Elige p de máximo descenso.
    """
    return -nd.Gradient(f)(x)
```

```python
def newton_dir(f,x):
    """ Elige p la dirección de Newton.
    """
    H = nd.Hessian(f)(x)
    return np.linalg.solve(H,-nd.Gradient(f)(x))
```

## Pruebas

### Función de Rosenbrok

```python
for m in ['newton', 'max', 'descoor']:
    print(f'Ahora trabajando en el método {m}')
    resp = busqueda_linea(rosen, np.array([2,3]), direction=m)
    n = resp['n_iter']
    x = resp['x*']
    print(f"tomó {n} iteraciones llegar a {x}")
          
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
for m in ['newton', 'max', 'descoor']:
    print(f'Ahora trabajando en el método {m}')
    resp = busqueda_linea(funcuad, np.array([5,5,5,5]), direction=m)
    n = resp['n_iter']
    x = resp['x*']
    print(f"tomó {n} iteraciones llegar a {x}")
    print()
          
```

```python

```
