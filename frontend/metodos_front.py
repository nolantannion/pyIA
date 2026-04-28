import numpy as np

def ajuste_lineal(x, y, sigmay = None):
    if sigmay is not None:
        coef, cov = np.polyfit(x, y, 1, w = 1/sigmay, cov=True)

    else:
        coef, cov = np.polyfit(x, y, 1, cov=True)

    m, b = coef
    sigma_m = np.sqrt(cov[0,0])
    sigma_b = np.sqrt(cov[1,1])

    y_fit = m * x + b

    return m, b, sigma_m, sigma_b, y_fit