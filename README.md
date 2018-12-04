# [HPCA] Projet : transport de neutron
### Clément Chouteau

[Lien vers les slides de la présentation de la partie 1](https://www.overleaf.com/read/ctsmpdsfxnjx)

## Compiler

- `make ` : compile toutes les versions.
- `make neutron-seq` : compile la version séquentielle CPU.
- `make neutron-omp` : compile la version multithread CPU, utilise OpenMP.
- `make neutron-gpu` : compile la version parallèle GPU, utilise CUDA.
- `make neutron-hyb` : compile la version hybride OpenMP + CUDA.

## Autres options du Makefile

- `make test` : compile toutes les versions et execute les tests.
- `make save` : compile toutes les versions avec enregistrement du résultat.
- `make clean` : efface tous les binaires.

## Optimiser

- `python optimizer.py 1.0 300000000 0.5 0.5` : trouve les meilleurs paramètres GPU (nombre de threads par bloc et nombre de neutrons par thread) qui minimisent le temps d'exécution.
- `python ratio.py 1.0 300000000 0.5 0.5 32 20000` : trouve le ratio neutrons à traiter sur CPU / GPU recommandé pour les paramètres courants.
