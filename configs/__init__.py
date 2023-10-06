from core import MetaObject
from .EnvironmentConfig import EnvironmentConfig
from .GraphicsConfig import GraphicsConfig
from .SimulationConfig import SimulationConfig
from .TrainConfig import TrainConfig


def createConfigs(configOverrides):
    """
    Utilitaire pour creer une config complete
    """

    try:
        configOverrides = MetaObject.from_json(configOverrides)
    except:
        configOverrides = None

    # creer les configs par defaut
    envConfig = EnvironmentConfig()
    gfxConfig = GraphicsConfig()
    simConfig = SimulationConfig()
    trainConfig = TrainConfig()

    # appliquer les overrides
    if not configOverrides is None:
        MetaObject.override_from_object(envConfig,
                                        configOverrides.environment)

        MetaObject.override_from_object(gfxConfig,
                                        configOverrides.graphics)

        MetaObject.override_from_object(simConfig,
                                        configOverrides.simulation)

        MetaObject.override_from_object(trainConfig,
                                        configOverrides.train)

    return MetaObject.from_kwargs(environment=envConfig,
                                  graphics=gfxConfig,
                                  simulation=simConfig,
                                  train=trainConfig)
