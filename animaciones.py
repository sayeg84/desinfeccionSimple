
import principal
import imageio
from matplotlib import animation
import matplotlib.pyplot as plt

np = principal.np
os = principal.os



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
def makeAnimation(gel_pos,vir_pos,vir_visits,colors=None,radius = None,savename="test.mp4",lifeFunc=principal.linearLife,fps=30):
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
            alphas = [lifeFunc(vir_visits[t,i],m=40) for i in range(n_vir)]
            # graficamos los viruses
            plt.scatter(vir_pos[:,0],vir_pos[:,1],s=100,c="black",alpha=alphas)
            for k in range(n_gel):
                # graficamos la estela de la trayectoria de un gel
                plt.plot(gel_pos[k,tail(t),0],gel_pos[k,tail(t),1],alpha=0.4,color=colors[k])
                # graficamos un gel
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
    with imageio.get_writer(savename, mode='I',fps=fps) as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join("temp_frames",filename))
            writer.append_data(image)
            os.remove(os.path.join("temp_frames",filename))

# correr la simulacion sin cambiarle nada
n_gel = 10
t_steps = 1000
mean_r = 0.01
n_vir = 50

radius = 0.002*np.random.randn(n_gel,t_steps) + mean_r
gel_pos, vir_pos, vir_visits, radius = principal.simulation(n_gel=n_gel,n_vir=n_vir,t_steps=t_steps,advanceFunc=principal.normalBoundary,radius = radius,radiusDec=True)
makeAnimation(gel_pos,vir_pos,vir_visits,radius=radius,savename=os.path.join("figures","test.mp4"))

"""
print("Anim 2")

radius = 0.005*np.random.randn(n_gel,t_steps) + mean_r
gel_pos, vir_pos, vir_visits, radius = principal.simulation(n_gel=n_gel,t_steps=t_steps,n_vir=n_vir,advanceFunc=principal.normalBoundary,radius = radius,radiusDec=False)
makeAnimation(gel_pos,vir_pos,vir_visits,radius=radius,savename=os.path.join("figures","normal.mp4"))

print("Anim 3")

radius = mean_r*np.ones((n_gel,t_steps))
gel_pos, vir_pos, vir_visits, radius = principal.simulation(n_gel=n_gel,t_steps=t_steps,n_vir=n_vir,advanceFunc=principal.normalBoundary,radius = radius,radiusDec=False)
makeAnimation(gel_pos,vir_pos,vir_visits,radius=radius,savename=os.path.join("figures","fixedR.mp4"))

print("Anim 4")

radius = 0.005*np.random.randn(n_gel,t_steps) + mean_r
gel_pos, vir_pos, vir_visits, radius = principal.simulation(n_gel=n_gel,t_steps=t_steps,n_vir=n_vir,advanceFunc=principal.reflectiveBoundary,radius = radius,radiusDec=True)
makeAnimation(gel_pos,vir_pos,vir_visits,radius=radius,savename=os.path.join("figures","reflective.mp4"))

print("Anim 5")

radius = 0.005*np.random.randn(n_gel,t_steps) + mean_r
gel_pos, vir_pos, vir_visits, radius = principal.simulation(n_gel=n_gel,t_steps=t_steps,n_vir=10000,advanceFunc=principal.normalBoundary,radius = radius,radiusDec=True)
makeAnimation(gel_pos,vir_pos,vir_visits,radius=radius,savename=os.path.join("figures","lowdensity.mp4"))
"""