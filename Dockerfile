FROM python:3.8-slim 
RUN apt-get update
RUN apt-get install -y ffmpeg curl git zsh
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
WORKDIR /main
COPY requirements.txt .
RUN pip install -r requirements.txt
ENV PYTHONUNBUFFERED=1
COPY . .
ENTRYPOINT [ "python" ]
CMD [ "main.py" ]