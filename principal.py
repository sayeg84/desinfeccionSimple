import imageio
from matplotlib import animation,rc 
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.spatial import kdtree
from scipy.integrate import odeint

"""
TPfriccion2D(V,t)

Función que define la ecuación diferencial asociada a un tiro parabólico con fricción en 2D
"""
def TPfriccion2D(V,t): # cte de gravedad, viscosidad de aire a 25°, densidad y diametro
    g = 9.81 #depende del tamaño de la gota
    mu=0.0000185
    rho=1050
    D=0.00035
    k=18*mu/(rho*D**2)
    v = np.sqrt(V[1]**2 + V[3]**2)
    dx = V[1]
    dvx = -k*V[1]*v
    dy = V[3]
    dvy = -k*V[3]*v-g
    return np.array([dx,dvx,dy,dvy]) #nos regresa las variables que definimos

"""
TPfriccion3D(V,t)

Función que define la ecuación diferencial asociada a un tiro parabólico con fricción en 2D
"""
def TPfriccion3D(v_x,t,g=9.81,mu=0.0000185,rho=1050,D=0.00035):#definimos la función
    dx=v_x[1]#definimos las variables 
    dvx= (-3*np.pi*D*mu*v_x[0]*6/rho*np.pi*(D**3))#definimos las variables con sus operaciones
    dy=v_x[3]
    dvy= (-3*np.pi*D*mu*v_x[2]*6/rho*np.pi*(D**3))
    dz=v_x[5]
    dvz= (-3*np.pi*D*mu*v_x[4]*6/rho*np.pi*(D**3))-g
    return np.array([dx,dvx,dy,dvy,dz,dvz])


"""
getIndex(sol,lim=1.0)

Dado un arreglo `sol`  que tiene las posiciones y velocidades solución a la ecuación de un tirpo parabólico, encuentra el índice de las posiciones entes de que la coordenada x exceda el valor de  `lim`.

Si no existe tal valor, regresa -1
"""
def getIndex(sol,lim=1.0):
    nt = sol.shape[0]
    for t in range(nt):
        if sol[t,0] > lim:
            return t-1
    return -1

"""
getInitialPositions(n,h=1.46,max_iter=1000)

Función que genera `n` condiciones iniciales para una persona con altura `h` para un máximo de iteraciones

Las condiciones iniciales se obtienen de intentar `max_iter` tiros parabólicos.
"""
def getInitialPositions(n,h=1.46,max_iter=1000):
    t = np.linspace(0,3,100)
    pos = np.zeros((n,2))
    # numero de tiros parabolicos exitosos que llevo
    p = 0
    # numero de tiros parabolicos totales
    c = 0
    while p < n and c < max_iter:
        c = c + 1
        # magnitud entre [50,50+10]
        r = 2 + 1*np.random.rand() # es para que no rebase los 2 m distancia
        # angulo polar (entre el eje z y la velocidad) entre [np.pi/6,np.pi/6+np.pi/6]
        theta = np.pi/8 + np.pi/8*np.random.rand() #es para que no rebase los 1.5 m de anchura
        # angulo azimutal (Respecto al ecuador) entre [-np.pi/4,np.pi/4]
        phi = (2*np.random.rand()-1)*np.pi/4
        dx = r*np.sin(theta)*np.cos(phi)
        dy = r*np.sin(theta)*np.sin(phi)
        dz = r*np.cos(theta)
        cond_ini = [0,dx,0,dy,h,dz] #ponemos las condiciones de las gráfica para los ejes
        sol = odeint(TPfriccion3D,cond_ini,t)
        # obtenemos el indice de cuando llega a un metro de distancia
        index = getIndex(sol)
        # guardamos la posicion solo en caso de que si llegue hasta 1m el tiro y la posicion z sea arriba de 1m
        if index > 0 and sol[index,4]>0.5:
            pos[p,:] = sol[index,[2,4]]
            p = p + 1
    return pos


"""
plotCircle(x,y,r,**kwargs):

Grafica un círculo de radio `r` con centro en `(x,y)`. 
Se puede darle color y otras propiedades a través del argumento `kwargs`.
"""
def plotCircle(x,y,r,**kwargs):
    # 101 puntos para el círculo
    angs = np.linspace(0,2*np.pi,101)
    xs = [x + r*np.cos(a) for a in angs]
    ys = [y + r*np.sin(a) for a in angs]
    plt.plot(xs,ys,**kwargs)


"""
tail(t,m=20)

Regresa un rango desde `0` hasta `t-1` si t<m y  uno desde `t-m` hasta `t` si t >= m. 

Se utiliza para graficar las estelas del gel que se mueve
"""
def tail(t,m=20):
    if t < m:
        return range(t)
    else:
        return range(t-m,t)


########################################

# Funciones de vida

########################################

"""
linearLife(x,m=60):

Función que de `x` que describe la línea recta que pasa por los puntos (0,1)  y (0,m).

Para x <0, regresa 1 y para x > m regresa 0
"""
def linearLife(x,m=60):
    if x <= 0:
        return 1
    elif x <= m:
        return 1-x/m
    else:
        return 0

"""
exponentialLife(x,in_high=60,exp=1.2):

Función de `x` que interpola de forma exponencial entre los puntos (0,1) y (in_high,0). `exp < 1` describe una linea logarítmica mientras que `exp` > 1 es una linea exponencial. `exp`=1 es una recta
"""
def exponentialLife(x,in_high=60,exp=1.2):
    in_low = 0
    out_low = 1
    out_high = 0
    if x <= in_low:
        return out_low 
    elif in_low < x < in_high: 
        if ((x-in_low)/(in_high-in_low)) > 0:
            return out_low + (out_high-out_low) * ((x-in_low)/(in_high-in_low))**exp 
        else:
            return  out_low + (out_high-out_low) * -((((-x+in_low)/(in_high-in_low)))**(exp))
    else: 
        return out_high

"""
normalBoundary(pos,v,L=[[0,0],[1,1]],r=0)

Función para evolucionar con condiciones normales a la frontera: nada sucede si alguna partícula llega a la frontera.
"""
def normalBoundary(pos,v,L=[[0,0],[1,1]],r=0):
    new_pos = pos + v
    return new_pos


"""
periodicBoundary(pos,v,L=[[0,0],[1,1]],r=0)

Función para evolucionar con condiciones periódicas a la frontera una partícula de gel en posición `pos` y velocidad `v`. `L` describe las dimensiones de la caja que contiene al gel y `r` el radio del gel. 
"""
def periodicBoundary(pos,v,L=[[0,0],[1,1]],r=0):
    new_pos = pos + v
    return np.array([L[0,i] + ((new_pos[i]-L[0,i]) % (L[1,i]-L[0,i])) for i in range(2)])


"""
reflectiveBoundary(pos,v,L=[[0,0],[1,1]],r=0)

Función para evolucionar con condiciones reflejantes a la frontera una partícula de gel en posición `pos` y velocidad `v`. `L` describe las dimensiones de la caja que contiene al gel y `r` el radio del gel. 
"""
def reflectiveBoundary(pos,v,L=[[0,0],[1,1]],r=0):
    new_pos = pos + v
    # iteramos sobre cada coordenada
    for i in range(pos.shape[0]):
        # checamos si excede el maximo de la caja
        if new_pos[i] + r > L[1,i]:
            # encontramos el tiempo al que se excede
            new_t = (L[1,i] - pos[i]-r)/v[i]
            # lo rebotamos
            new_pos[i] = pos[i] + new_t*v[i] - (1-new_t)*v[i]
        # checamos si excede el minimo de la caja
        elif new_pos[i] - r < L[0,i]:
            # encontramos el tiempo al que se excede
            new_t = (r + L[0,i] - pos[i])/v[i]
            # lo rebotamos
            new_pos[i] = pos[i] + new_t*v[i] - (1-new_t)*v[i]
    return new_pos

"""
simulation(n_gel=20,n_vir=60,t_steps=1000,advanceFunc=periodicBoundary,radius=None)

Funcion para hacer la simulación. Tiene los siguientes argumentos:

* n_gel: numero de partículas de gel
* n_vir: numero de particulas de virus
* t_steps: pasos de la simulacion
* advanceFunc: funcion para avanzar la simulacion. Debe ser `periodicBoundary`, `reflectiveBoundary` o `normalBoundary`
* radius: arreglo de tamaño `n_gel` que representa los radios de las partículas de gel

La funcion regresa una tupla de tres arreglos multidimensionales:

* gel_pos: (dimension n_gel x t_steps x dim). Las posiciones de las particulas de gel para todos los tiempos.
* vir_pos: (dimension n_vir x dim) las posiciones de las particulas de virus.
* vir_visits: (dimension n_vir  x t_steps)  las visitas a cada partícula de virus en cada tiempo.
"""
def simulation(n_gel=20,n_vir=60,t_steps=1000,advanceFunc=periodicBoundary,radius = None, radiusDec=False):
    print("Making simulation")
    # variable auxiliar para guarda la dimension de la simulacion
    dim=2
    # si los radios de los geles no están definidos, hacerlos 0.1 por default
    if radius is None:
        radius = 0.1*np.ones((n_gel,t_steps))
    # generar las condiciones iniciales de los viruses
    vir_pos = getInitialPositions(n_gel)
    # obenter su minimo y su maximo para definir las dimensiones de la caja
    # que contiene a las particulas de gel
    mins = vir_pos.min(axis=0)
    maxs = vir_pos.max(axis=0)
    # prealocando arreglo para las visitas a los viruses como funcion del tiempo y del numero de 
    vir_visits = np.zeros((t_steps,n_vir))
    # prealocando arreglo para 
    gel_pos = np.zeros((n_gel,t_steps,dim))
    # creando las posiciones iniciales del gel de forma aleatoria
    # uniformemente distribuida entre el minimo y el maximo de las
    # posiciones en x y en y del gel
    gel_pos[:,0,:] = (maxs-mins)*np.random.rand(n_gel,dim) + mins
    # arbol de búsqueda para buscar eficientemente si un virus está cerca de un gel
    tree = kdtree.KDTree(vir_pos)
    # funcion auxiliar para avanzar la simulación
    func = lambda x,y : advanceFunc(x,y,L=np.array([mins,maxs]))
    # iteramos sobre el tiempo
    for t in range(1,t_steps):
        vir_visits[t,:] = vir_visits[t-1,:]
        # iteramos sobre los geles
        for k in range(n_gel):
            # Buscamos todos los viruses vecinos a ese gel
            neighs = tree.query_ball_point(gel_pos[k,t-1,:],radius[k,t-1])
            # Le añadimos una visita a ese virus
            vir_visits[t,neighs] +=  1
            # Generamos una magnitud y dirección aleatoria para mover el gel
            r = 0.05*np.random.rand()
            theta = 2*np.pi*np.random.rand()
            # generamos una velocidad para esa magnitud y direccion
            v = r*np.array([np.cos(theta),np.sin(theta)])
            # avanzamos el gel
            gel_pos[k,t,:] = func(gel_pos[k,t-1,:],v)
        # decrece el radio para hacerlo realista
        if radiusDec:
            radius[:,t] =  radius[:,t-1] - radius[:,0]/t_steps
    return gel_pos, vir_pos, vir_visits, radius




"""
makeAnimation(gel_pos,vir_pos,vir_visits,colors=None,radius = None,name="test.mp4",lifeFunc=linearLife,fps=30)

Funcion para hacer la animación de una simulación. Tiene los argumentos:

* gel_pos: lo mismo que `gel_pos` de la función `simulation`.
* vir_pos: lo mismo que `vir_pos` de la función `simulation`.
* gel_pos: lo mismo que `vir_visits` de la función `simulation`.
* colors: un arreglo de longitud `n_gel` con identificadores de colores.
* radius: un arreglo de longitud `n_gel` con los valores de los radios.
* name: un string con el nombre y la extensión en la cual guardar la animación.
* lifeFunc: la función que nos da la vida de un virus como función del número de visitas.
* fps: numero de cuadros por segundo para el video 

"""
def makeAnimation(gel_pos,vir_pos,vir_visits,colors=None,radius = None,name="test.mp4",lifeFunc=linearLife,fps=30,savename="test"):
    # obtenemos el numero de particulas de gel y de pasos de tiempo
    n_gel,t_steps = gel_pos.shape[0:2]
    # numero de viruses
    n_vir = vir_visits.shape[1]
    # auxiliar para el formateo de archivos
    n_pad = int(np.floor(np.log10(t_steps)))+1
    # radios default
    if radius is None:
        radius = 0.1*np.ones((n_gel,t_steps))
    # colores default
    if colors is None:
        colors = ["C{0}".format(k) for k in range(n_gel)]
    # valores minimos o  maximos
    mins = vir_pos.min(axis=0)
    maxs = vir_pos.max(axis=0)
    # frecuencia de cuadros para poner en el gif
    freq = 3
    print("Making frames")
    for t in range(t_steps):
        if t % freq ==0:
            # figura para hacer la gráfica
            fig = plt.figure()
            plt.grid()
            # los alphas (la intensidad del color de los viruses) se dibujan según cuanta vida tengan
            alphas = [linearLife(vir_visits[t,i],m=40) for i in range(n_vir)]
            # graficamos los viuses
            plt.scatter(vir_pos[:,0],vir_pos[:,1],s=100,c="black",alpha=alphas)
            for k in range(n_gel):
                # graficamos la estela de la trayectoria de un virus
                plt.plot(gel_pos[k,tail(t),0],gel_pos[k,tail(t),1],alpha=0.4,color=colors[k])
                # graficamos un virus
                plotCircle(gel_pos[k,t,0],gel_pos[k,t,1],radius[k,t],alpha=0.7,color=colors[k])
            # limites de la grafica
            plt.xlim(mins[0],maxs[0])
            plt.ylim(mins[1],maxs[1])
            plt.title("time = {0}".format(t))
            # forzamos el aspect ratio a 1
            plt.gca().set_aspect('equal', 'box')
            # guardamos la grafica
            plt.tight_layout()
            plt.savefig(os.path.join("temp_frames","{0}.png".format(str(t).zfill(n_pad))),dpi=300)
            plt.close()
    # hacemos la animación
    print("Making animation")
    filenames = sorted(os.listdir("temp_frames"))
    with imageio.get_writer(name, mode='I',fps=fps) as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join("temp_frames",filename))
            writer.append_data(image)
            os.remove(os.path.join("temp_frames",filename))


if __name__ == "__main__":
    # correr la simulacion sin cambiarle nada
    n_gel = 100
    t_steps = 1000
    mean_r = 0.05
    radius = 0.005*np.random.randn(n_gel,t_steps) + mean_r
    gel_pos, vir_pos, vir_visits, radius = simulation(n_gel=n_gel,t_steps=t_steps,n_vir=500,advanceFunc=normalBoundary,radius = radius,radiusDec=True)
    makeAnimation(gel_pos,vir_pos,vir_visits,radius=radius)