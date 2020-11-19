import sys
import importlib

loaded_modules = set()
def FreezeModules():
    global loaded_modules
    loaded_modules = set(sys.modules)

def Reimport():
    global loaded_modules
    to_reload = set(sys.modules) - loaded_modules

    for module in to_reload:
        if not sys.modules[module]:
            continue

        try:
            if "site-packages" in sys.modules[module].__file__:
                continue
        except:
            continue

        importlib.reload(sys.modules[module])