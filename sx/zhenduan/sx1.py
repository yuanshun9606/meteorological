import math

def f_e(T):
    x = 17.269 * (T - 273.16) / (T - 35.86)
    e = 6.1078 * pow(math.e, x)
    return e

def f_q(e, P):
    return 0.622 * e / (P - 0.378 * e)

def main():
    RH, T, P = map(float, input().split())
    es = f_e(T)
    e = RH * es
    Td0 = T
    # 求Td
    while (True):
        e0 = f_e(Td0)
        if (e < e0):
            Td0 -= 0.05
        else:
            Td = Td0
            break
    print('Td=', Td - 273.16)
    # 求ZL TL PL
    ez = f_e(Td)
    thetaz = T * pow((1000 / P), 0.286)
    qZ = f_q(ez, P)
    PL = 300
    while (True):
        TL = thetaz*pow((1000/PL),-0.286)
        eL = f_e(TL)
        qL = f_q(eL,PL)
        if qL>=qZ:
            break
        else:
            PL += 1
    ZL= (9.8*35.2+1004*(T-TL))/9.8
    print('ZL=',ZL,'TL=',TL-273.16,'PL=',PL)

main()
