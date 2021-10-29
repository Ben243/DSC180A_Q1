FROM ucsdets/datascience-notebook:2021.2-stable

USER root

RUN apt-get -y install htop tmux

USER jovyan

RUN pip install --no-cache-dir pandas numpy matplotlib