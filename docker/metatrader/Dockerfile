FROM continuumio/miniconda3

EXPOSE 80

WORKDIR /usr/share/metatraderai

# Create the environment:
COPY environment.yml /usr/share/metatraderai/environment.yml
WORKDIR /usr/share/metatraderai

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

#RUN conda config --set channel_priority strict
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "traderenv", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"
#RUN pip install opencv-python
#RUN pip install autorom[accept-rom-license]
#RUN pip install gym[atari]



RUN python -c "import cv2"
# The code to run when container is started:



COPY src .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "traderenv", "python", "Trainning_Code/metarun.py","-p","80"]
