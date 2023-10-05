from core import MetaObject
from .GameConfig import GameConfig
from .GraphicsConfig import GraphicsConfig
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
    envConfig = GameConfig()
    gfxConfig = GraphicsConfig()
    trainConfig = TrainConfig()

    # appliquer les overrides
    if not configOverrides is None:
        MetaObject.override_from_object(envConfig,
                                        configOverrides.environment)

        MetaObject.override_from_object(gfxConfig,
                                        configOverrides.graphics)

        MetaObject.override_from_object(trainConfig,
                                        configOverrides.train)

    return MetaObject.from_kwargs(environment=envConfig,
                                  graphics=gfxConfig,
                                  train=trainConfig)
