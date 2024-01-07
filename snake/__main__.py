"""
Encapsule les options du command line pour demarrer l'application dans le mode desire.
"""


import click
import os

from snake.core import MetaObject, RandomProxy


# enlever le message d'init de pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
pass_config = click.make_pass_decorator(MetaObject)


@click.group()
@click.pass_context
def cli(ctx):
    from .configs import configsCreate
    ctx.obj = configsCreate("config_overrides.json")

@cli.command()
@click.option("windowSize",
              "-w",
              type=int,
              help="Taille de la fenêtre d'affichage en pixels.")
@click.option("-fps",
              type=int,
              help="Vitesse du serpent en frame par seconde")
@click.option("record",
              "-r",
              type=str,
              help="Nom de fichier pour enregistrer des parties. Doit inclure le chemin. % sera remplacer par "
                   "le numéro de partie. Ex: recordings/game_%.json")
@click.option("recordN",
              "-rn",
              type=int,
              help="Si -r est spécifié, enregistre un épisode tout les N parties.")
@click.option("agent",
              "-a",
              type=str,
              help="Type de l'agent à utiliser.")
@pass_config
def play(configs, windowSize, fps, record, recordN, agent):
    "Lance le jeu en mode interactif"
    import ai.agents as agents
    from application import ApplicationInteractive
    from application.wrappers.ai.agents import AgentActionRecorder, AgentInteractive

    if not windowSize is None and windowSize > 0:
        configs.graphics.windowSize = windowSize

    if not fps is None and fps > 0:
        # on veut faire le rendu a 60 fps mais ralentir
        # la simulation: determiner le diviseur en consequence
        divider = int(configs.graphics.fps / fps + 0.5)
        divider = max(divider, 1)
        configs.graphics.simulationFpsDivider = divider

    application = ApplicationInteractive(configs)

    if not agent is None:
        # instantier un agent a partir d'un string
        # limiter aux classes de ai.agents pour le moment
        agent_class = getattr(agents, agent)

        if not agent_class is AgentInteractive:
            application.agent = agent_class(configs.train, configs.simulation)

    if not record is None:
        application.window.caption += " - recording"
        application.agent = AgentActionRecorder(application.agent, record, recordN)

    application.runAttended()

    if not record is None:
        application.agent.save()

@cli.command()
@click.option("windowSize",
              "-w",
              type=int,
              help="Taille de la fenêtre d'affichage en pixels.")
@click.option("-fps",
              type=int,
              help="Frame par seconde désiré")
@click.argument("recording", type=click.Path(exists=True))
@pass_config
def replay(configs, windowSize, fps, recording):
    "Rejoue RECORDING en mode non interactif"
    from application import ApplicationInteractive
    from application.wrappers.ai.agents import AgentActionPlayback

    if not windowSize is None and windowSize > 0:
        configs.graphics.windowSize = windowSize

    if not fps is None and fps > 0:
        # on veut faire le rendu a un fps specifique
        configs.graphics.fps = fps
        configs.graphics.simulationFpsDivider = 1

    configs.graphics.caption += " - playback"

    application = ApplicationInteractive(configs)
    application.agent = AgentActionPlayback(recording)
    application.runAttended()

@cli.command()
@click.option("windowSize",
              "-w",
              type=int,
              help="Taille de la fenêtre d'affichage en pixels.")
@click.option("-fps",
              type=int,
              help="Frame par seconde désiré")
@click.argument("recording", type=click.Path(exists=True))
@pass_config
def render(configs, windowSize, fps, recording):
    "Convertie RECORDING dans un fichier mp4"
    from application import ApplicationInteractive
    from application.wrappers.ai.agents import AgentActionPlayback
    from application.wrappers.graphics import VideoWriter

    configs.graphics.showWindow = False
    configs.graphics.simulationFpsDivider = 1

    if not windowSize is None and windowSize > 0:
        configs.graphics.windowSize = windowSize

    if not fps is None and fps > 0:
        configs.graphics.fps = fps

    # override window pour avoir enregistrement video
    filename, _ = os.path.splitext(recording)
    filename = f"{filename}.mp4"

    application = ApplicationInteractive(configs)
    application.agent = AgentActionPlayback(recording)
    application.window = VideoWriter(application.window,
                                     configs.graphics.fps,
                                     filename)
    application.runUnattended()
    application.window.dispose()

@cli.command()
@click.option("windowSize",
              "-w",
              type=int,
              help="Taille de la fenêtre d'affichage.")
@click.option("-fps",
              type=int,
              help="Frame Par Seconde de l'affichage.")
@click.option("record",
              "-r",
              type=str,
              help="Nom de fichier pour enregistrer les épisodes. Inclue le chemin. % sera remplacer par "
                   "le numéro d'épisode. Le format est toujours json. Ex: recordings/train_%.json")
@click.option("recordN",
              "-rn",
              type=int,
              help="Si -r est spécifié, enregistre un épisode tout les N parties.")
@click.option("unattended",
              "-u",
              is_flag=True,
              default=False,
              help="Train sans rendu.")
@click.option("episodes",
              "-e",
              type=int,
              help="Nombre d'épisodes pour l'entrainement.")
@click.option("episodeMaxLen",
              "-eml",
              type=int,
              help="Longueur maximale pour un épisode.")
@click.option("agent",
              "-a",
              type=str,
              help="Type de l'agent à utiliser.")
@pass_config
def train(configs,
          windowSize,
          fps,
          record,
          recordN,
          unattended,
          episodes,
          episodeMaxLen,
          agent):
    "Entraine un agent"
    import ai.agents as agents

    from application import ApplicationTrain
    from application.wrappers.ai.agents import AgentActionRecorder

    configs.train.unattended = unattended

    if not episodes is None and episodes > 0:
        configs.train.episodes = episodes

    if not episodeMaxLen is None and episodeMaxLen > 0:
        configs.train.episodeMaxLen = episodeMaxLen

    if not agent is None and len(agent) > 0:
        configs.train.agent = agent

    if not windowSize is None and windowSize > 0:
        configs.graphics.windowSize = windowSize

    if not fps is None and fps > 0:
        configs.environment.renderFps = fps

    agent = configs.train.agent if agent is None else agent

    # instantier un agent a partir d'un string
    # limiter aux classes de ai.agents pour le moment
    agent_class = getattr(agents, agent)
    agent = agent_class(configs.train, configs.simulation)

    if not record is None:
        configs.graphics.caption += " - recording"
        agent = AgentActionRecorder(agent, record, recordN)

        path, _ = os.path.splitext(record)
        record = f"{path}.csv"

    ApplicationTrain(configs, agent, record).run()

    if not record is None:
        agent.save()

if __name__ == "__main__":
    # mettre le repertoire courant comme celui par defaut
    # (facilite la gestion des chemins relatifs)
    path = os.path.abspath(__file__)
    path, _ = os.path.split(path)
    os.chdir(path)

    # initialiser random
    RandomProxy.init()

    cli()
