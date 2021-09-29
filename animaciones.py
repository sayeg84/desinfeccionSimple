import os
import principal
np = principal.np
# correr la simulacion sin cambiarle nada
n_gel = 100
t_steps = 1000
mean_r = 0.05
n_vir = 500

print("Anim1")

radius = 0.005*np.random.randn(n_gel,t_steps) + mean_r
gel_pos, vir_pos, vir_visits, radius = principal.simulation(n_gel=n_gel,t_steps=t_steps,n_vir=n_vir,advanceFunc=principal.normalBoundary,radius = radius,radiusDec=True)
principal.makeAnimation(gel_pos,vir_pos,vir_visits,radius=radius,savename=os.path.join("figures","radiusDecLinear.mp4"))

print("Anim 2")

radius = 0.005*np.random.randn(n_gel,t_steps) + mean_r
gel_pos, vir_pos, vir_visits, radius = principal.simulation(n_gel=n_gel,t_steps=t_steps,n_vir=n_vir,advanceFunc=principal.normalBoundary,radius = radius,radiusDec=False)
principal.makeAnimation(gel_pos,vir_pos,vir_visits,radius=radius,savename=os.path.join("figures","normal.mp4"))

print("Anim 3")

radius = mean_r*np.ones((n_gel,t_steps))
gel_pos, vir_pos, vir_visits, radius = principal.simulation(n_gel=n_gel,t_steps=t_steps,n_vir=n_vir,advanceFunc=principal.normalBoundary,radius = radius,radiusDec=False)
principal.makeAnimation(gel_pos,vir_pos,vir_visits,radius=radius,savename=os.path.join("figures","fixedR.mp4"))

print("Anim 4")

radius = 0.005*np.random.randn(n_gel,t_steps) + mean_r
gel_pos, vir_pos, vir_visits, radius = principal.simulation(n_gel=n_gel,t_steps=t_steps,n_vir=n_vir,advanceFunc=principal.reflectiveBoundary,radius = radius,radiusDec=True)
principal.makeAnimation(gel_pos,vir_pos,vir_visits,radius=radius,savename=os.path.join("figures","reflective.mp4"))

print("Anim 5")

radius = 0.005*np.random.randn(n_gel,t_steps) + mean_r
gel_pos, vir_pos, vir_visits, radius = principal.simulation(n_gel=n_gel,t_steps=t_steps,n_vir=10000,advanceFunc=principal.normalBoundary,radius = radius,radiusDec=True)
principal.makeAnimation(gel_pos,vir_pos,vir_visits,radius=radius,savename=os.path.join("figures","lowdensity.mp4"))