import principal

np = principal.np
plt = principal.plt
os = principal.os

if not os.path.isdir("figures"):
    os.mkdir("figures")

xs = np.linspace(0,60,101)
exps = [0.1,0.5,1.0,3.0,8.0]

fig = plt.figure()
for ex in exps:
    ys = [principal.exponentialLife(x,exp=ex) for x in xs]
    plt.plot(xs,ys,label=f"ex={ex}")
plt.grid()
plt.ylabel("Vitalidad")
plt.xlabel("Número de visitas")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("figures","vitalidad.pdf"))
plt.close()

ns = range(20,51,1)
m = 20
ts = 500

vals = np.zeros((len(ns),m))

fig = plt.figure()
for ex in exps:
    lifeFunction = lambda x : principal.exponentialLife(x,exp=ex)
    for i,k in enumerate(ns):
        radius = 0.05*np.ones((k,ts))
        for l in range(m):
            print(f"Advance: {(i*m+l)*100/(m*len(ns))} %")
            position, obs_pos, obs_visits, radius = principal.simulation(n_gel=k,t_steps=ts,radius=radius)
            vals[i,l] = np.mean([lifeFunction(x) for x in obs_visits[-1,:]])
    plt.errorbar(ns,np.mean(vals,axis=1),
                yerr = np.std(vals,axis=1),
                fmt="o",
                ecolor="black",
                capsize=4,
                label=f"ex={ex}")
    np.savetxt(os.path.join("data",f"{ex}-m.csv"),vals,delimiter=",")
plt.xlabel("número de caminantes")
plt.ylabel("Vitalidad media final ")
plt.grid()
plt.legend()
plt.savefig(os.path.join("figures","n_gel.pdf"))
plt.close()
"""
"""

rs = np.linspace(0.01,0.1,41)
k = 40
vals = np.zeros((len(rs),m))
fig = plt.figure()


for ex in exps:
    lifeFunction = lambda x : principal.exponentialLife(x,exp=ex)
    for i,r in enumerate(rs):
        radius = r*np.ones((k,ts))
        for l in range(m):
            print(f"Advance: {(i*m+l)*100/(m*len(rs))} %")
            position, obs_pos, obs_visits, radius = principal.simulation(n_gel=k,t_steps=ts,radius=radius)
            vals[i,l] = np.mean([lifeFunction(x) for x in obs_visits[-1,:]])
    
    plt.errorbar(rs,np.mean(vals,axis=1),
                yerr = np.std(vals,axis=1),
                fmt="o",
                ecolor="black",
                capsize=4,
                label=f"ex={ex}")
    np.savetxt(os.path.join("data",f"{ex}-r.csv"),vals,delimiter=",")
plt.xlabel("Radio")
plt.ylabel("Vitalidad media final")
plt.grid()
plt.legend()
plt.savefig(os.path.join("figures","r.pdf"))
plt.close()