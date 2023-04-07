# Recuperação de Informação
Trabalhos de Recuperação de Informação - 2023/1.

## Setup virtual environments
```
python3.11 -m venv pa1 

source pa1/bin/activate

pip3.11 install -r requirements.txt
```

## Run
```
python3 indexer.py -m <MEMORY> -c <CORPUS> -i <INDEX>
```
where:
* `-m <MEMORY>`: memory available to the indexer in megabytes.
* `-c <CORPUS>`: path to the corpus file to be indexed.
* `-i <INDEX>`: path to the directory where indexes should be written.
