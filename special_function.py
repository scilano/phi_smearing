import mpmath as mp
#Compute the lerch-hankel function, shamelessly copied from https://fredrikj.net/blog/2022/02/computing-the-lerch-transcendent/
def lerch_hankel(z, s, a):
    z, s, a = mp.mpmathify(z), mp.mpmathify(s), mp.mpmathify(a)
    if z == 0 or z == 1 or mp.isint(a):
        return mp.nan
    if mp.re(a) < 2:
        return z * lerch_hankel(z, s, a + 1) + a**(-s)
    g = lambda t: t**(s-1)*mp.exp(-a*t)/(1-z*mp.exp(-t))
    h = lambda t: (-t)**(s-1)*mp.exp(-a*t)/(1-z*mp.exp(-t))
    L = mp.log(z)
    if mp.isint(s) and mp.re(s) >= 1:
        if abs(mp.im(L)) < 0.25 and mp.re(L) >= 0:
            if mp.im(z) <= 0.0:
                I = mp.quad(g, [0, +1j, +1j+abs(L)+1, abs(L)+1, mp.inf])
            else:
                I = mp.quad(g, [0, -1j, -1j+abs(L)+1, abs(L)+1, mp.inf])
        else:
            I = mp.quad(g, [0,mp.inf])
        return mp.rgamma(s) * I
    if mp.re(L) < -0.5:
        residue = 0
        c = min(abs(mp.re(L)) / 2, 1)
        left = right = top = c
    elif abs(mp.im(L)) > 0.5:
        residue = 0
        c = min(abs(mp.im(L)) / 2, 1)
        left = right = top = c
    else:
        residue = (-L)**s / L / z**a
        left = max(0, -mp.re(L)) + 1
        top = abs(mp.im(L)) + 1
        right = abs(L) + 1
    isreal = (mp.im(z) == 0) and (mp.re(z) < 1) and (mp.im(s) == 0) and (mp.im(a) == 0) and (mp.re(a) > 0)
    w = mp.mpc(-1)**(s-1)
    I = 0
    if isreal:
        I += 2j * mp.im(mp.quad(g, [right, right + top * 1j]) / w)
        I += 2j * mp.im(mp.quad(g, [right + top * 1j, -left + top * 1j]) / w)
        I += 2j * mp.im(mp.quad(h, [-left + top * 1j, -left]))
        I += mp.quad(g, [right, mp.inf]) * (w - 1/w)
    else:
        I += mp.quad(g, [right, right + top * 1j]) / w
        I += mp.quad(g, [right + top * 1j, -left + top * 1j]) / w
        I += mp.quad(h, [-left + top * 1j, -left - top * 1j])
        I += mp.quad(g, [-left - top * 1j, right - top * 1j]) * w
        I += mp.quad(g, [right - top * 1j, right]) * w
        I += mp.quad(g, [right, mp.inf]) * (w - 1/w)
    I = I / (2*mp.pi*1j) + residue
    I = -mp.gamma(1-s) * I
    return I