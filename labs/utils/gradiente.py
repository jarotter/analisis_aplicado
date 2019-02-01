# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Gradiente
#
# Sólo usamos métodos con precisión $O(h)$.

import numpy as np

def gradiente(f, x, h=10e-6, method='forward'):
    """
    Aproximación numérica al gradiente de una función f:R^n -> R
    
    Entradas:
    ---------
    f : función
        La función a derivar.
    x : ndarray
        El punto donde interesa evaluar el gradiente.
    h : double
        Tamaño de la diferencia
    method: string
        Una de 'forward', 'backward' o 'central', qué diferencia finita usar.
        
    Regresa:
    --------
    grad : ndarray
        El gradiente de f en x.
    """
    z = np.ones_like(x)
    
    if method == 'forward':
        return (f(x+h*z)-f(x))/h
    elif method == 'backward':
        return (f(x)-f(x-h*z))/h
    elif method == 'central':
        return (f(x+h*z)-f(x-h*z))/h
    else:
        raise ValueError("¿Qué clase de método es ese?")
