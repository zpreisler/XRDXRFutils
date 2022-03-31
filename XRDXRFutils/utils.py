from numpy import minimum,fft,pad
from scipy.signal import windows

def snip(z,m):

    x = z.copy()
    for p in range(1,m)[::-1]:
        a1 = x[p:-p]
        a2 = (x[:(-2 * p)] + x[(2 * p):]) * 0.5
        x[p:-p] = minimum(a2,a1)

    return x

def convolve(z,n=21,std=3):
    kernel = windows.gaussian(2 * n - 1,std)

    y_pad = pad(z,(n,n),'edge')

    f = fft.rfft(y_pad)
    w = fft.rfft(kernel,y_pad.shape[-1])
    y = fft.irfft(w * f)

    return y[n * 2:] / sum(kernel)

def snip2d(z,m):

    x = z.copy()
    for p in range(1,m)[::-1]:
        a1 = x[:,p:-p]
        a2 = (x[:,:(-2 * p)] + x[:,(2 * p):]) * 0.5
        x[:,p:-p] = minimum(a2,a1)

    return x

def convolve2d(z,n=21,std=3):
    kernel = windows.gaussian(2 * n - 1,std)

    y_pad = pad(z,((0,0),(0,0),(n,n)),'edge')

    f = fft.rfft(y_pad)
    w = fft.rfft(kernel,y_pad.shape[-1])
    y = fft.irfft(w * f)

    return y[:,n * 2:] / sum(kernel)

def snip3d(z,m):

    x = z.copy()
    for p in range(1,m)[::-1]:
        a1 = x[:,:,p:-p]
        a2 = (x[:,:,:(-2 * p)] + x[:,:,(2 * p):]) * 0.5
        x[:,:,p:-p] = minimum(a2,a1)

    return x

def convolve3d(z,n=21,std=3):
    kernel = windows.gaussian(2 * n - 1,std)

    y_pad = pad(z,((0,0),(0,0),(n,n)),'edge')

    f = fft.rfft(y_pad)
    w = fft.rfft(kernel,y_pad.shape[-1])
    y = fft.irfft(w * f)

    return y[:,:,n * 2:] / sum(kernel)
