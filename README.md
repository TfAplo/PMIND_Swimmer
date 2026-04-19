# Reinforcement Learning — TD3 & Action Horizon Extension

> Projet de M1 — Exploration du Reinforcement Learning avec TD3, wrappers d'observation/action, et visualisations avancées sur PointMaze.

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture du projet](#architecture-du-projet)
- [Algorithme : TD3](#algorithme--td3)
- [Wrappers](#wrappers)
- [Environnements](#environnements)
- [Visualisations](#visualisations)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure des fichiers de sortie](#structure-des-fichiers-de-sortie)
- [Résultats](#résultats)
- [Dépendances principales](#résultats)

---

## Vue d'ensemble

Ce projet explore l'algorithme **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** dans des environnements de contrôle continu. Il a été développé avec la bibliothèque [BBRL](https://github.com/osigaud/bbrl) et suit plusieurs axes de recherche :

1. **Baseline TD3** — implémentation standard sur l'environnement Swimmer (MuJoCo)
2. **Observabilité partielle** — utilisation de wrappers pour masquer ou historiser des features d'observation
3. **Prédiction d'horizon d'action (M actions)** — l'agent prédit une séquence de M actions mais n'exécute que la première, via un `ActionTimeExtensionWrapper`
4. **Analyse sur PointMaze** — passage sur un environnement plus simple et plus interprétable pour visualiser et comprendre le comportement appris

---

## Architecture du projet

```text
.
├── main.py                     # Script principal d'entraînement (CLI avec Argparse)
├── src/
│   ├── agent.py                # Définition des acteurs et critiques
│   ├── config.py               # Paramètres et configurations
│   ├── env_maze_2D.py          # Environnements Maze2D personnalisés
│   ├── TD3.py                  # Classe TD3 et boucle d'apprentissage
│   ├── loss.py                 # Fonctions de coût (MSE, Actor loss)
│   ├── wrapper.py              # Wrappers d'extension et de filtrage
│   ├── visualisation.py        # Outils de tracés (Heatmaps, Trajectoires)
│   └── integrated_gradient.py  # Analyse d'importance des actions
├── experiments/                # Notebooks et scripts de recherche initiaux
```

---

## Algorithme : TD3

L'implémentation suit la variante **EpochBased** de BBRL. Les composantes principales sont :

**Acteur** (`ContinuousDeterministicActor`) — Réseau déterministe avec activation Tanh en sortie, qui produit `M × action_dim` valeurs représentant la séquence d'actions future.

**Critiques** (`ContinuousQAgent` × 2) — Deux Q-réseaux indépendants. La cible utilise le minimum des deux pour réduire la surestimation (clipped double Q-learning).

**Bruit d'exploration** (`AddGaussianNoise` ou `AddOUNoise`) — Ajout de bruit pour l'exploration continue.

**Stabilisations TD3 classiques :**
- Policy delay : l'acteur et les target networks sont mis à jour 1 fois sur `policy_delay` (par défaut : 2)
- Target policy smoothing : bruit gaussien clippé ajouté aux actions cibles
- Soft updates des target networks (coefficient τ)

**Hyperparamètres principaux :**

| Paramètre | Valeur |
|---|---|
| Buffer size | 1 000 000 |
| Batch size | 256 |
| Discount γ | 0.99 |
| Learning rate | 3e-4 |
| τ (soft update) | 0.005 |
| Action noise (exploration) | 0.1 |
| Target policy noise | 0.2 |
| Policy delay | 2 |
| Max epochs | 400 000 |

---

## Wrappers

### `ActionTimeExtensionWrapper`

L'idée centrale du projet : l'agent apprend à prédire **M actions simultanément**, mais seule la première est effectivement exécutée dans l'environnement. L'espace d'action est donc étendu d'un facteur M.

```python
class ActionTimeExtensionWrapper(gym.Wrapper):
    """
    Étend l'espace d'action d'un facteur M.
    L'agent produit M * action_dim valeurs, mais seule action[0] est jouée.
    """
```

L'hypothèse testée : forcer l'agent à planifier sur un horizon court l'amène à apprendre des représentations plus riches, notamment utiles dans des contextes partiellement observables.

---

### `ObsTimeExtensionWrapper`

Construit un buffer glissant des `size` dernières observations et les concatène. Utile pour donner une "mémoire" à l'agent dans des environnements partiellement observables.

```python
class ObsTimeExtensionWrapper(gym.ObservationWrapper):
    """
    Concatène les (size + 1) dernières observations.
    Les observations manquantes au début de l'épisode sont remplacées par des zéros.
    """
```

---

### `FeatureFilterWrapper`

Masque un index d'observation donné, simulant une perte d'information sensorielle (observabilité partielle).

```python
class FeatureFilterWrapper(gym.ObservationWrapper):
    """
    Supprime la feature à l'index `idx` de l'espace d'observation.
    """
```

---

### `VelocityControlWrapper`

Modifie la dynamique de l'environnement pour permettre à l'agent de contrôler directement sa vitesse, en ignorant l'inertie physique du modèle. L'action générée est ajustée via un multiplicateur (velocity_multiplier). (Option `--ignore_inertia`)

```python
class VelocityControlWrapper(gym.Wrapper):
    """
    Applique un contrôle direct de la vitesse en ignorant l'inertie.
    Les actions de l'agent sont multipliées par `velocity_multiplier` 
    pour définir la vélocité immédiate.
    """
```

---

## Environnements

### Swimmer (MuJoCo)
Utilisé pour les expériences initiales de TD3 standard et les tests avec observabilité partielle.

### PointMaze (`MazeMur`)

Environnement principal pour l'analyse du comportement. Un robot ponctuel doit naviguer dans un labyrinthe avec un mur central ou un U central, du départ `r` jusqu'à l'objectif `g`. Exemple du mur central :

```
1 1 1 1 1 1 1 1 1 1 1
1 . . . . . . . . . 1
1 . . . . 1 . . . . 1
1 . . . . 1 . . . . 1
1 . r . . 1 . . g . 1
1 . . . . 1 . . . . 1
1 . . . . 1 . . . . 1
1 . . . . . . . . . 1
1 1 1 1 1 1 1 1 1 1 1
```

L'observation est aplatie via `FlattenObservation` puis étendue par `ActionTimeExtensionWrapper`.

---

## Visualisations

Trois types de visualisations sont générées automatiquement toutes les **5 000 steps**, pour comprendre les décisions de l'agent :

### 1. Heatmap V(s) + Vector Field

Pour chaque cellule libre du labyrinthe, on calcule la valeur estimée `V(s) = Q(s, π(s))` et on superpose les flèches d'action prédites par l'acteur. Avec M > 1, les M actions successives sont représentées avec un dégradé de couleur.

![Exemple vector field](outputs/M=1/MazeMur/.../heatmap_vector_fields/vector_field_step400000.png)

### 2. Heatmap V(s) + Trajectoire réelle

L'agent joue un épisode complet depuis la position de départ. La trajectoire est tracée par-dessus la heatmap, avec les flèches d'intention espacées pour ne pas surcharger l'image.

### 3. Integrated Gradients — Importance des actions

Pour M > 1, on utilise les **Integrated Gradients** pour mesurer l'importance que le critique accorde à chacune des M actions dans la séquence. Une courbe d'évolution est sauvegardée et mise à jour à chaque checkpoint.

```
ig_history.json  →  {"steps": [...], "a0": [...], "a1": [...], ...}
```

### Enregistrement Vidéo 

L'option `--visualize_best` permet d'enregistrer le comportement du meilleur modèle.

---

## Installation

```bash
# Cloner le repo
git clone [https://github.com/TfAplo/PMIND_Swimmer.git](https://github.com/TfAplo/PMIND_Swimmer.git)
cd PMIND_Swimmer

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install torch numpy gymnasium scipy omegaconf matplotlib tqdm
pip install bbrl bbrl-gymnasium bbrl-utils gymnasium-robotics
```

> **Note :** MuJoCo est requis pour l'environnement Swimmer. Pour PointMaze seul, `gymnasium-robotics` suffit.

---

## Utilisation

Le script `main.py` utilise `argparse` pour configurer facilement les entraînements.

### Exemple de lancement :

#### Lancement simple sur PointMaze avec M=1 :

```bash
python main.py --env_type pointmaze --M 1 --plots
```

#### Lancement avec prédiction de 3 actions (M=3) en ignorant l'inertie :

```bash
python main.py --env_type pointmaze --M 3 --ignore_inertia --visualize_best
```

#### Entraînement sur Maze2D forme "U" avec multi-seeds en parallèle :

```bash
python main.py --env_type maze2D --maze_map U --nb_seeds 5 --multiprocess --save_checkpoint
```

### Arguments CLI principaux (`main.py`)

Le script s'utilise en ligne de commande avec de nombreux paramètres pour configurer l'expérience. 

**Environnement (Requis)**
* `--env_type` : L'environnement à utiliser. Choix : `maze2D` ou `pointmaze`.

**Paramètres généraux**
* `--M` : Taille de la séquence d'actions, si M=5, réalise 5 runs de M=1 à M=5 (défaut : `1`).
* `--nb_seeds` : Nombre de random seeds à exécuter (défaut : `5`).
* `--seed_start` : Numéro de la seed de départ (défaut : `0`).
* `--logdir` : Chemin pour log les résultats et Tensorboard (défaut : `logs`).
* `--vel_mult` : Multiplicateur de vitesse quand l'option `--ignore_inertia` est active (défaut : `10.0`).
* `--plots` : Affiche les heatmaps et les trajectoires à la fin de chaque run.

**Options de l'environnement et sauvegarde**
* `--maze_map` : Choix de la carte (uniquement pour `maze2D`). Choix : `wall` ou `U` (défaut : `wall`).
* `--save_checkpoint` : Sauvegarde les checkpoints des modèles (Actor/Critic) dans le dossier de log.
* `--visualize_best` : Enregistre une vidéo de la meilleure run à la fin de l'entraînement.
* `--ignore_inertia` : Active le `VelocityControlWrapper` pour ignorer l'inertie et utiliser directement la vitesse au lieu de l'accélération.

**Exécution**
* `--multiprocess` : Utilise le multiprocessing Python pour lancer les différentes seeds en parallèle sur plusieurs cœurs de votre machine.

---

## Structure des fichiers de sortie

Vos résultats seront stockés dans le dossier `output/` à la racine du projet (au même niveau que `main.py`). L'architecture sépare les logs Tensorboard des visualisations et modèles :

```text
output/
├── tblogs/
│   └── <env_name>/
│       └── <datetime>_td3-S<seed>_M=<M>/
│           └── events.out.tfevents...      # Fichiers Tensorboard (Reward, Critic/Actor Loss)
└── <env_name>/
    └── <datetime>_td3-S<seed>_M=<M>/
        ├── heatmap_vector_field/           # Visualisations : Heatmaps + champs de vecteurs
        ├── heatmap_real_traj/              # Visualisations : Heatmaps + trajectoires réelles
        └── checkpoints/                    # (Si --save_checkpoint) Checkpoints des modèles
```

Pour visualiser l'entraînement en temps réel avec Tensorboard, pointez vers le sous-dossier `tblogs` :

```bash
tensorboard --logdir output/tblogs/
```

---

## Résultats

Les principales questions explorées dans ce projet :

- **Observabilité partielle** : `ObsTimeExtensionWrapper` + `FeatureFilterWrapper` compensent-ils une perte d'information ?
- **M = 1 vs M > 1** : Prédire plusieurs actions améliore-t-il la politique apprise, même si seule la première est jouée ?
- **Integrated Gradients** : les actions lointaines dans la séquence sont-elles exploitées par le critique, ou l'agent apprend-il à les ignorer ?

Les visualisations sur PointMaze permettent de répondre qualitativement à ces questions en observant la cohérence des vecteurs d'action avec la géométrie du labyrinthe.

---

## Dépendances principales

- [PyTorch](https://pytorch.org/)
- [BBRL](https://github.com/osigaud/bbrl) — framework RL modulaire
- [Gymnasium](https://gymnasium.farama.org/) / [gymnasium-robotics](https://robotics.farama.org/)
- [OmegaConf](https://omegaconf.readthedocs.io/)
- [Matplotlib](https://matplotlib.org/)

---

*Projet réalisé dans le cadre du Projet MIND — M1 MIND - Sorbonne Université*