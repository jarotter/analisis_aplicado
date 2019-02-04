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
from scipy.optimize import rosen
```

# Laboratorio 3 - Búsqueda de línea por interpolación.


Sea $f:\mathbb{R}^n \to \mathbb{R}$ continuamente diferenciable en $\mathbf{x}$ y $\mathbf{p}$ una dirección de descenso de $f$ en $\mathbf{x}$. Definimos $g: [0,1] \to \mathbb{R}$ como $g(t)=f(\mathbf{x}+t\mathbf{p})$. La condición de Armijo (W1) puede escribirse como

$$
g(t) \leq \ell(1)
$$

donde $\ell(t) = f(\mathbf{x})+\left(\nabla f(\mathbf{x})'\mathbf{p}\right)c_1t$. 

Supongamos que se inicia la búsqueda del paso óptimo en $t=1$. Si $g(1) \leq \ell(1)$, se satisface la condición y terminamos. En este laboratorio construimos un método para el caso $g(1) > \ell(1)$.

## Método 
Si $g(1) > \ell(1)$, construimos un polinomio cuadrático $q \in \mathcal{P}_2(t)$, digamos $q(t)=c+bt+at^2$ tal que

1. $g(0)=q(0)$
2. $g(1)=q(1)$
3. $g'(0) = q'(0)$.

### Ejercicio
Muestre que $q$ tiene un mínimo en $(0,1)$.


*Demostración.*

Es fácil ver que el polinomio interpolador tiene $a=g(1)-g(0)-g'(0)$. Luego, como $g(1)>\ell(1)$,

$$a>l(1)-g(0)-g'(0) = g(0)+c_1g'(0)-g(0)-g'(0) = g'(0)(c_1-1)$$.

Como $0<c_1<1$, $a>0$ y el polinomio $q$ es continuo y estrictamente convexo en el compacto $[0,1]$, por lo que tiene un mínimo. Derivando e igualando a cero, el mínimo se da en 

$$
t^*=\frac{-b}{2a} = \frac{-\nabla f(\mathbf{x})'\mathbf{p}}{2\left(f(\mathbf{x}+\mathbf{p})-f(\mathbf{x})-\nabla f(\mathbf{x})'\mathbf{p}\right)}
$$

Como $\nabla f(\mathbf{x})'\mathbf{p}<0$, el mínimo no se da en cero.
Supongamos que el mínimo fuera $q(1)$. Intuitivamente, como estamos fijando los extremos de la parábola y ya nos aseguramos que no coincide con la recta que conecta $(0,g(0)$ con $(1, g(1))$ (porque $a>0$), la parábaola debe forzosamente pasar por debajo de la recta en algunos puntos.


## Laboratorio

Programe el método de búsqueda de línea en la dirección de Newton con interpolación cuadrática.

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

### Pruebas

Por ahora están en el laboratorio 1. 
> TODO: Pensar en cómo ir juntando todo.
