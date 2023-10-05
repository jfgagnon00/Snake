from core.MetaObject import MetaObject


def createConfigs(config_overrides):
    """
    Utilitaire pour creer une config complete
    """

    try:
        config_overrides = MetaObject.from_json(config_overrides)
    except:
        config_overrides = None

    from configs.GameConfig import GameConfig
    from graphics.GraphicsConfig import GraphicsConfig

    # creer les configs par defaut
    env_config = GameConfig()
    gfx_config = GraphicsConfig()

    # appliquer les overrides
    if not config_overrides is None:
        MetaObject.override_from_object(env_config,
                                        config_overrides.environment)

        MetaObject.override_from_object(gfx_config,
                                        config_overrides.graphics)

    return MetaObject.from_kwargs(environment=env_config,
                                  graphics=gfx_config)
