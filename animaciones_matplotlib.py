import principal
from matplotlib import animation
import matplotlib.pyplot as plt

np = principal.np
os = principal.os

"""
plotCircleUpdate(line,x,y,r,**kwargs):

Grafica un circulo con centro `x,y` y radio `r` sobre el objeto `line` de matplotlib.

"""
def plotCircleUpdate(line,x,y,r,**kwargs):
    # 101 puntos para el círculo
    angs = np.linspace(0,2*np.pi,101)
    xs = [x + r*np.cos(a) for a in angs]
    ys = [y + r*np.sin(a) for a in angs]
    line.set_data(xs,ys,**kwargs)

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




#exps = [0.1,0.5,1.0,3.0,8.0]
exps=[1.0]
for b in [True,False]:
    for ex in exps:
        # parámetros de simulacion
        name = f"highDensity-{b}.mp4"
        lifeFunc = lambda x : principal.exponentialLife(x,exp=ex)
        n_gel = 1000
        n_vir = 5000
        t_steps = 2000
        freq = 6
        mean_r = 0.005
        radius = 0.0008*np.random.randn(n_gel,t_steps) + mean_r


        gel_pos, vir_pos, vir_visits, radius = principal.simulation(n_gel=n_gel,n_vir=n_vir,t_steps=t_steps,advanceFunc=principal.normalBoundary,radiusDec=False,radius=radius)
        colors = None
        n_gel,t_steps = gel_pos.shape[0:2]
        
        # numero de viruses
        n_vir = vir_visits.shape[1]
        
        # colores default
        if colors is None:
            colors = ["turquoise".format(k) for k in range(n_gel)]
        
        # valores minimos o  maximos
        mins = vir_pos.min(axis=0)
        maxs = vir_pos.max(axis=0)

        # graficas de circulos y colas para actualizar
        circles = []
        tails = []
        
        fig = plt.figure()
        ax = plt.axes(xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
        points = ax.scatter([], [],s=10)
        plt.gca().set_aspect('equal', 'box')
        plt.grid()
        plt.tight_layout()



        """
        init()

        Inicializacion de la animacion
        """
        def init():
            points.set_offsets(vir_pos)
            points.set_facecolors("black")
            for k in range(n_gel):
                circles.append(ax.plot([],[],alpha=0.7,color=colors[k]))
                #tails.append(ax.plot([],[],alpha=0.7,color=colors[k],lw=0.4))

        def animate(t):
            alphas = [lifeFunc(vir_visits[t,i]) for i in range(n_vir)]
            points.set_alpha(alphas)
            for k in range(n_gel):
                # graficamos la estela de la trayectoria de un gel
                #tails[k][0].set_data(gel_pos[k,tail(t),0],gel_pos[k,tail(t),1])
                # graficamos un gel
                plotCircleUpdate(circles[k][0],gel_pos[k,t,0],gel_pos[k,t,1],radius[k,t])
            ax.set_title(f"step = {t}")

        print("Making animation")
        anim = animation.FuncAnimation(fig,animate,frames=range(0,t_steps,freq),interval=10,init_func=init)
        print("Saving animation")
        anim.save(os.path.join("figures",name),fps=30,dpi=300)
        plt.close()