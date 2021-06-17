import pygmt

fig = pygmt.Figure()
# Make a global Mollweide map with automatic ticks
fig.basemap(region="g", projection="W15c", frame=True)
# Plot the land as light gray, and the water as sky blue
fig.coast(land="#666666", water="skyblue")
fig.savefig("test.jpg")
