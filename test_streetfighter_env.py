import retro

try:
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
    obs = env.reset()
    env.render()
    env.close()
    print("Environment setup successful!")
except AttributeError as e:
    print("Error:", e)
